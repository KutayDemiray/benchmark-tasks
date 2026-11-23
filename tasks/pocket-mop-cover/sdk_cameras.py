"""Render three stock cameras from native SDK states without feeding poses to physics."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np


def author_products(stage, width, height):
    """Author three calibrated RGB products using the installed OVRTX device contract."""
    from pxr import UsdGeom, Gf, Sdf
    cameras = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
    assert len(cameras) == 3
    cameras_by_name = {}
    for name, suffix in (('top_cam', '/TopCam'), ('left_cam', '/left_wrist_cam'), ('right_cam', '/right_wrist_cam')):
        matches = [p for p in cameras if p.endswith(suffix)]
        assert len(matches) == 1
        cameras_by_name[name] = matches[0]
        intrinsics=stage.GetPrimAtPath(matches[0]).GetAttribute('omni:lensdistortion:opencvPinhole:imageSize')
        assert intrinsics and tuple(intrinsics.Get())==(width,height), 'Use the stock camera calibration at the requested resolution'
        # This installed kitless USD distribution has no UsdRender Python
        # wrapper. Author the standard typed prims/properties directly.
        product = stage.DefinePrim('/TaskCameraProducts/'+name, 'RenderProduct')
        product.CreateAttribute('resolution', Sdf.ValueTypeNames.Int2, custom=False).Set(Gf.Vec2i(width, height))
        product.CreateRelationship('camera', custom=False).SetTargets([matches[0]])
        variable = stage.DefinePrim('/TaskCameraProducts/'+name+'/LdrColor', 'RenderVar')
        variable.CreateAttribute('sourceName', Sdf.ValueTypeNames.String, custom=False).Set('LdrColor')
        product.CreateRelationship('orderedVars', custom=False).SetTargets([variable.GetPath()])
        # The installed RTX API selects one masked CUDA device and maps it to Vulkan.
        product.CreateAttribute('deviceIds', Sdf.ValueTypeNames.UIntArray, custom=False).Set([0])
        product.CreateAttribute('omni:rtx:waitForEvents', Sdf.ValueTypeNames.TokenArray, custom=True).Set(['AllLoadingFinished', 'OnlyOnFirstRequest'])
        product.CreateAttribute('omni:rtx:rendermode', Sdf.ValueTypeNames.Token, custom=True).Set('RealTimePathTracing')
    return cameras_by_name, {name: '/TaskCameraProducts/'+name for name in cameras_by_name}


class SDKCameras:
    """Own a detached RTX visualization of the native physics state."""

    def __init__(self, session, robot, cloth_path, *, width=640, height=360, rigid_sources=()):
        from gpu_coordination import require_reservation
        from pxr import Usd, UsdGeom, Gf
        import ovrtx
        self.assignment = require_reservation()
        self.sim = session
        self.robot = robot
        self.width,self.height=width,height
        assert (width,height) in ((640,360),(320,176))
        self.pose_sources=[*robot.arms,*rigid_sources]
        self.cloth_path = cloth_path
        self.renderer = None
        self.writers = {}
        self.count = 0
        self.ovrtx = ovrtx
        stage = Usd.Stage.Open(str(session.out/'scene.usda'))
        self.cameras, self.products = author_products(stage, width, height)
        self.paths = []
        cache = UsdGeom.XformCache()
        for arm in self.pose_sources:
            for name in arm['poses'].body_names:
                path = arm['paths'][name]
                scale = np.asarray(Gf.Transform(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).GetScale())
                assert np.allclose(scale, 1., atol=1e-6, rtol=0), 'Unexpected stock visual scale'
                self.paths.append(path)
        render_scene = session.out/'render-scene.usda'
        stage.GetRootLayer().Export(str(render_scene))
        config = ovrtx.RendererConfig(active_cuda_gpus='0',
            log_file_path=str(session.out/'ovrtx.log'), log_level='info', read_gpu_transforms=False,
            keep_system_alive=False, suppress_deprecation_warnings=True)
        # Reuse the installed renderer's settings-extension adapter, without patching it.
        # It loads the exact config before renderer construction and fails closed.
        from isaaclab_ov.renderers.ovrtx_shader_cache import _acquire_settings_applier
        apply_setting = _acquire_settings_applier(config)
        assert apply_setting is not None, 'Installed OVRTX settings extension is required'
        shader_cache = session.out/'shader-cache'
        shader_cache.mkdir()
        settings = {'/renderer/activeGpu': self.assignment['index'],
                    '/renderer/multiGpu/enabled': 'false', '/renderer/multiGpu/activeCudaGpus': '0,',
                    '/rtx/shaderDb/driverShaderCachePath': str(shader_cache),
                    '/rtx/shaderDb/driverAppShaderCachePath': str(shader_cache)}
        for key, value in settings.items():
            assert apply_setting('--'+key+'='+str(value)), 'OVRTX rejected setting '+key
        self.renderer = ovrtx.Renderer(config)
        self.renderer.open_usd(str(render_scene))
        self.renderer.write_attribute(self.paths+[cloth_path], 'omni:resetXformStack',
                                      np.ones(len(self.paths)+1, dtype=np.bool_))
        self.renderer.write_attribute([cloth_path], 'omni:xform', np.eye(4, dtype=np.float64)[None],
                                      semantic=ovrtx.Semantic.XFORM_MAT4x4)
        self.provenance = dict(scene_sha256=hashlib.sha256((session.out/'scene.usda').read_bytes()).hexdigest(),
            render_scene_sha256=hashlib.sha256(render_scene.read_bytes()).hexdigest(),
            camera_paths=self.cameras, resolution=[width, height], frames_per_second=20,
            renderer='installed ovrtx', physical_gpu=self.assignment,
            active_cuda_gpus='0', render_product_cuda_device_ids=[0], single_device_requested=True, accepted_renderer_settings=settings,
            physics_pose_writes=False, source='Native SDK link poses and cloth positions at each control boundary.',
            visual_stage_separate_from_physics=True, camera_intrinsics_and_mounts_from_source_scene=True)

    def update(self, points):
        from scipy.spatial.transform import Rotation
        poses = np.concatenate([self.sim.read(arm['poses'])[0] for arm in self.pose_sources])
        matrices = np.tile(np.eye(4, dtype=np.float64), (len(poses), 1, 1))
        matrices[:, :3, :3] = Rotation.from_quat(poses[:, 3:]).as_matrix().transpose(0, 2, 1)
        matrices[:, 3, :3] = poses[:, :3]
        self.renderer.write_attribute(self.paths, 'omni:xform', matrices, semantic=self.ovrtx.Semantic.XFORM_MAT4x4)
        self.renderer.write_array_attribute([self.cloth_path], 'points', [np.asarray(points, dtype=np.float32)])

    def render(self):
        return self.renderer.step(render_products=set(self.products.values()), delta_time=1/20)

    def warmup_and_verify(self, points, physics_binding):
        import torch
        from gpu_coordination import scheduler_binding
        self.update(points)
        for _ in range(5):
            products = self.render()
        product = products[self.products['top_cam']]
        mapping = product.frames[0].render_vars['LdrColor'].map(device=self.ovrtx.Device.CUDA)
        try:
            mapping.wait()
            rgba = torch.from_dlpack(mapping)
            native_render_device = str(rgba.device)
            image_shape = list(rgba.shape)
        finally:
            del rgba
            mapping.unmap()
        native = torch.empty(physics_binding.shape, dtype=torch.float32, device='cuda:0')
        physics_binding.read(native)
        xml = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=5)
        allocations = []
        for device in ET.fromstring(xml).findall('gpu'):
            for process in device.findall('processes/process_info'):
                if process.findtext('pid') == str(os.getpid()):
                    allocations.append(dict(uuid=device.findtext('uuid'), type=process.findtext('type'),
                        memory_mib=float(process.findtext('used_memory').split()[0])))
        text = (self.sim.out/'ovrtx.log').read_text(errors='replace')
        active = sorted(set(int(value) for value in re.findall(r'^\|\s*(\d+)\s*\|[^\n]*Yes:', text, re.MULTILINE)))
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        gpu = self.assignment
        checks = dict(native_physics=str(native.device)=='cuda:0', native_render=native_render_device=='cuda:0',
            render_image_shape=image_shape == [self.height, self.width, 4],
            uuid=uuid.removeprefix('GPU-').lower()==gpu['uuid'].removeprefix('GPU-').lower(),
            actual_renderer=active == [gpu['index']],
            observed_compute_graphics=any(p['uuid']==gpu['uuid'] and 'C' in p['type'] and 'G' in p['type'] for p in allocations),
            no_other_device_work=all(p['uuid']==gpu['uuid'] or (p['type']=='G' and p['memory_mib']<=16) for p in allocations))
        report = dict(pid=os.getpid(), job_id=os.environ.get('ROBOCURVE_JOB_ID'), expected_index=gpu['index'],
            expected_uuid=gpu['uuid'], cuda_uuid=uuid, physics_tensor_device=str(native.device),
            renderer_enabled=True, native_render_tensor_device=native_render_device,
            active_renderer_physical_indices=active, active_cuda_gpus='0', allocations=allocations,
            native_api='ovphysx TensorBinding.read and ovrtx LdrColor CUDA output', checks=checks, verified=all(checks.values()))
        if 'ROBOCURVE_JOB_ID' in os.environ:
            scheduler_binding()
            from scheduler import identity, write_json
            report['process_identity'] = identity(os.getpid())
            write_json(Path(os.environ['ROBOCURVE_JOB_DIR'])/'device-verification.json', report)
        (self.sim.out/'gpu-verification.json').write_text(json.dumps(report, indent=2))
        assert report['verified'], 'SDK physics/render assignment verification failed: '+str(checks)
        (self.sim.out/'sdk-camera-provenance.json').write_text(json.dumps(self.provenance, indent=2))

    def snapshot(self, points):
        """Render current measured state without advancing physics or recording an action."""
        self.update(points)
        products = self.render()
        images = {}
        for name, path in self.products.items():
            product = products[path]
            mapping = product.frames[0].render_vars['LdrColor'].map(device=self.ovrtx.Device.CPU)
            try:
                mapping.wait()
                rgba = np.from_dlpack(mapping)
                rgb = np.ascontiguousarray(rgba[:, :, :3]).copy()
                assert rgb.dtype == np.uint8 and rgb.shape == (self.height, self.width, 3)
                images[name] = rgb
            finally:
                del rgba
                mapping.unmap()
        return images

    def record(self, points):
        from PIL import Image
        images = self.snapshot(points)
        self.count += 1
        for name, rgb in images.items():
            if name not in self.writers:
                self.writers[name] = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo',
                    '-pix_fmt', 'rgb24', '-s', f'{self.width}x{self.height}', '-r', '20', '-i', '-', '-c:v', 'libx264',
                    '-threads', '2', '-preset', 'ultrafast', '-crf', '23', '-pix_fmt', 'yuv420p',
                    str(self.sim.out/(name+'.mp4'))], stdin=subprocess.PIPE)
            self.writers[name].stdin.write(rgb.tobytes())
            if self.count%20 == 0:
                Image.fromarray(rgb).save(self.sim.out/(name+'-latest.png'))
        return images

    def close(self):
        for writer in self.writers.values():
            writer.stdin.close()
            assert writer.wait() == 0, 'Camera encoder failed'
        self.writers.clear()
        if self.renderer is not None:
            self.renderer.destroy()
            self.renderer = None
        self.provenance['recorded_frames'] = self.provenance.get('shared_recorded_frames', self.count)
        self.provenance.setdefault('recording_owner', 'SDKCameras.record')
        (self.sim.out/'sdk-camera-provenance.json').write_text(json.dumps(self.provenance, indent=2))
