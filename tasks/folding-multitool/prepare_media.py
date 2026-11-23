"""CPU-only review sheets and anonymous H.264 media packaging; never uploads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import imageio_ffmpeg


def probe(path):
    # ffprobe is not installed. Use the already-installed imageio ffmpeg
    # binary for both inspection and transcoding; do not add dependencies.
    frames = imageio_ffmpeg.read_frames(str(path))
    try:
        return {**next(frames), "size_bytes": Path(path).stat().st_size}
    finally:
        frames.close()


def review_sheets(run):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw
    run = Path(run)
    frames = imageio.get_reader(run/"diagnostic.mp4")
    # Every simulated second, all three camera views; no temporal editing of
    # the eventual video. Labels appear only in these diagnostic review sheets.
    page, tiles, paths = 0, [], []
    destination = run/"review-sheets"
    destination.mkdir(exist_ok=True)
    def save():
        nonlocal page
        sheet = Image.new("RGB", (960, 12*108), (245, 245, 245))
        for j, (second, frame) in enumerate(tiles):
            tile = Image.fromarray(frame).resize((480, 88))
            x, y = (j % 2)*480, (j//2)*108
            sheet.paste(tile, (x, y+20))
            ImageDraw.Draw(sheet).text((x+4, y+3), f"{second:.1f} s", fill=(0, 0, 0))
        path = destination/f"{page:02d}.png"
        sheet.save(path)
        paths.append(str(path))
        page += 1
    try:
        for i, frame in enumerate(frames):
            if i % 20:
                continue
            tiles.append((i/20, np.asarray(frame)))
            if len(tiles) == 24:
                save()
                tiles = []
        if tiles:
            save()
    finally:
        frames.close()
    return {"sheets": paths, "video": probe(run/"diagnostic.mp4")}


def package(demo, model, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"uploaded": False, "media_id": None, "artifacts": {}}
    for kind, directory in (("demo", demo), ("model_run", model)):
        if directory is None:
            continue
        run = Path(directory).resolve()
        meta = json.loads((run/"metadata.json").read_text())
        result = json.loads((run/"result.json").read_text())
        rescore = json.loads((run/"independent-rescore.json").read_text())
        assert meta["mode"] == "retrieval" and rescore["passed"]
        assert meta["max_steps"] == 3600 and result["steps"] <= 3600
        if kind == "demo":
            assert meta["controller"] == "scripted" and result["scoring"]["score"] == 1.
            assert json.loads((run/"recording-review.json").read_text())["continuous_physical_demonstration_accepted"]
            shutil.copy2(run/"setup-all.png", destination/"setup.png")
        else:
            assert meta["controller"] == "dream" and meta["hosted"]
            transcript = json.loads((run/"inference-transcript.json").read_text())
            assert transcript["requests"] and not transcript["dry_run"]
        target = destination/(kind+".mp4")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-v", "error", "-i", str(run/"diagnostic.mp4"),
                        "-map_metadata", "-1", "-an", "-c:v", "libx264", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)], check=True)
        details = probe(target)
        assert details["codec"] == "h264"
        assert details["pix_fmt"].split("(")[0] == "yuv420p"
        assert target.stat().st_size < 50_000_000
        frames, duration = imageio_ffmpeg.count_frames_and_secs(str(target))
        assert frames == result["steps"]+1, "Recording must include the reset and every executed action"
        details.update(decoded_frames=frames, decoded_duration=duration)
        manifest["artifacts"][kind] = {"path": str(target.resolve()), "source_run": str(run),
            "score": result["scoring"]["score"], "steps": result["steps"], "termination": result["reason"],
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "probe": details}
    (destination/"media-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-run")
    parser.add_argument("--demo-run")
    parser.add_argument("--model-run")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.review_run:
        print(json.dumps(review_sheets(args.review_run), indent=2))
    else:
        if not args.out or not (args.demo_run or args.model_run):
            parser.error("Provide --review-run, or --out and --demo-run/--model-run")
        print(json.dumps(package(args.demo_run, args.model_run, args.out), indent=2))
