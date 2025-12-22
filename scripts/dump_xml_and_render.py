# dump_xml_then_run.py
import mujoco
import traceback

orig_from_xml_string = mujoco.MjModel.from_xml_string

def patched_from_xml_string(xml, *args, **kwargs):
    try:
        return orig_from_xml_string(xml, *args, **kwargs)
    except Exception:
        dump_path = "/tmp/robosuite_failed_model.xml"
        with open(dump_path, "w") as f:
            f.write(xml)
        print(f"[DEBUG] dumped failed xml to {dump_path}")
        traceback.print_exc()
        raise

mujoco.MjModel.from_xml_string = patched_from_xml_string

# 现在再 import 你的渲染脚本并执行
import scripts.test_offscreen_video as tov

if __name__ == "__main__":
    tov.main()
