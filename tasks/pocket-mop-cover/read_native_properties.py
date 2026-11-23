"""Read native object properties from a version-0.3/0.4 OmniPVD diagnostic capture.

The stream layout follows NVIDIA's PhysX 5.9.1 OmniPvdReaderImpl and public
OmniPvdCommands/Defines headers. This offline reader never starts simulation.
Large arrays are hashed; scalar/small-array first and last values are retained.
It fails on unknown commands, unsupported versions, and truncated records.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct


SOURCE = 'https://github.com/NVIDIA-Omniverse/PhysX/tree/110.3-omni-and-physx-5.9.1/physx/pvdruntime'
FORMATS = {0: 'b', 1: 'h', 2: 'i', 3: 'q', 4: 'B', 5: 'H', 6: 'I', 7: 'Q',
           8: 'f', 9: 'd', 11: 'Q', 12: 'I', 13: 'I'}


def decode_value(raw, spec, *, unlimited_array=False):
    """Decode a declared scalar/array without interpreting unrecorded properties."""
    kind = spec.get('type')
    if kind == 10:
        return raw.rstrip(b'\0').decode('utf-8', errors='replace')
    if kind == 13:
        assert len(raw) in (1, 2, 4, 8), 'Unsupported native flags storage'
        return dict(value=int.from_bytes(raw, 'little'), storage_bits=8*len(raw))
    if kind in FORMATS:
        code = FORMATS[kind]
        size = struct.calcsize('<'+code)
        assert len(raw) % size == 0, 'Misaligned OmniPVD attribute data'
        count = len(raw)//size
        if count <= 64 or unlimited_array:
            values = list(struct.unpack('<'+code*count, raw))
            return values[0] if count == 1 else values
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                hex=raw.hex() if len(raw) <= 64 else None)


def read_capture(path, *, array_properties=()):
    """Return the complete parsed stream inventory with first/last property values."""
    classes, attributes, objects, live = {}, {}, [], {}
    counts = Counter()
    frames, messages, unmatched_destroys = [], [], []
    with path.open('rb') as stream:
        def read(count):
            value = stream.read(count)
            assert len(value) == count, f'Truncated OmniPVD at byte {stream.tell()}'
            return value

        def unpack(code):
            values = struct.unpack('<'+code, read(struct.calcsize('<'+code)))
            return values[0] if len(values) == 1 else values

        def string():
            return read(unpack('H')).decode('utf-8', errors='replace')

        version = unpack('III')
        assert version in ((0, 3, 0), (0, 4, 0)), f'Unsupported OmniPVD version {version}'
        while True:
            command = stream.read(1)
            if not command:
                break
            command = command[0]
            counts[command] += 1
            if command == 1:
                handle, base = unpack('II')
                assert handle not in classes, 'Duplicate class handle'
                classes[handle] = dict(name=string(), base=base)
            elif command in (3, 4, 5):
                owner, handle = unpack('II')
                spec = dict(owner=owner, command=command)
                if command == 4:
                    spec['class_type'] = unpack('I')
                else:
                    spec['type'] = unpack('H')
                    if command == 3:
                        spec['count_or_enum'] = unpack('I')
                spec['name'] = string()
                assert handle not in attributes, 'Duplicate attribute handle'
                attributes[handle] = spec
            elif command in (6, 7, 8):
                context, handle = unpack('QQ')
                depth = unpack('B')
                assert depth > 0, 'Empty attribute path'
                handles = [unpack('I') for _ in range(depth)]
                raw = read(unpack('I'))
                if (context, handle) not in live:
                    obj = dict(context=context, handle=handle, class_name='UNRECORDED_CREATION',
                               class_handle=None, name='', properties={}, list_events=[], created=False)
                    objects.append(obj)
                    live[(context, handle)] = obj
                obj = live[(context, handle)]
                key = '.'.join(attributes[item]['name'] for item in handles)
                spec=attributes[handles[-1]]
                property_id=classes[spec['owner']]['name']+'.'+spec['name']
                value = decode_value(raw, spec, unlimited_array=property_id in array_properties)
                if command == 6:
                    prop = obj['properties'].setdefault(key, dict(first=value, updates=0))
                    prop.update(last=value, updates=prop['updates']+1)
                else:
                    obj['list_events'].append(dict(command=command, property=key, value=value))
            elif command == 9:
                context = unpack('Q')
                kind = unpack('I')
                handle = unpack('Q')
                name = string()
                obj = live.get((context, handle))
                if obj is None:
                    obj = dict(context=context, handle=handle, properties={}, list_events=[])
                    objects.append(obj)
                    live[(context, handle)] = obj
                else:
                    assert not obj.get('created', True), 'Reused live object handle'
                    obj['properties_precede_creation'] = True
                obj.update(class_name=classes[kind]['name'], class_handle=kind, name=name, created=True)
            elif command == 10:
                key = unpack('QQ')
                if key in live:
                    live.pop(key)['destroyed'] = True
                else:
                    unmatched_destroys.append(dict(context=key[0], handle=key[1]))
            elif command in (11, 12):
                context, timestamp = unpack('QQ')
                frames.append(dict(command=command, context=context, timestamp=timestamp))
            elif command == 13:
                context = unpack('Q')
                message, file = string(), string()
                line, kind, class_handle = unpack('III')
                messages.append(dict(context=context, message=message, file=file,
                                     line=line, type=kind, class_handle=class_handle))
            else:
                raise AssertionError(f'Unknown OmniPVD command {command} at byte {stream.tell()-1}')
        parsed_bytes = stream.tell()
    assert parsed_bytes == path.stat().st_size
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(source=str(path), sha256=digest, stream_version=version,
                format_reference=SOURCE, reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                parsed_all_bytes=True, parsed_bytes=parsed_bytes,
                command_counts=dict(counts), classes=classes, attributes=attributes,
                objects=objects, frames=frames, messages=messages, unmatched_destroys=unmatched_destroys,
                limitations=['Only properties actually published by the native OmniPVD integration are known.',
                             'Capture is diagnostic; it does not establish task success or solver equivalence.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = read_capture(args.capture)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(parsed_bytes=result['parsed_bytes'],
                          objects=len(result['objects']), frames=len(result['frames']),
                          classes=dict(Counter(obj['class_name'] for obj in result['objects'])),
                          messages=result['messages']), indent=2))


if __name__ == '__main__':
    main()
