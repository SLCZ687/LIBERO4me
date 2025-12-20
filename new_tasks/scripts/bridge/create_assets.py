import os

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 定义资产根目录
ASSET_ROOT = "custom_assets"
create_folder(ASSET_ROOT)

# ==========================================
# 1. Bridge Brick XML
# ==========================================
# solref: 0.01 (时间常数变大，接触变软)
# solimp: 0.95 0.99 0.001 (稍微降低刚性)
brick_xml_content = """
<mujoco model="bridge_brick">
  <asset>
    <texture name="tex_brick" type="cube" builtin="flat" rgb1="1 0.8 0.8" width="512" height="512"/>
    <material name="mat_brick" texture="tex_brick" shininess="0.5" specular="0.5"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom name="brick_geom" type="box" size="0.075 0.015 0.02" material="mat_brick" 
              density="800" friction="1.5 0.005 0.0001" solref="0.001 1" solimp="0.95 0.99 0.001"
              condim="4" group="1"/>
        <geom name="brick_vis" type="box" size="0.075 0.015 0.02" material="mat_brick" 
              contype="0" conaffinity="0" group="0"/>
        <site name="bottom_site" pos="0 0 -0.02" size="0.002" rgba="0 0 0 0"/>
        <site name="top_site" pos="0 0 0.02" size="0.002" rgba="0 0 0 0"/>
        <site name="horizontal_radius_site" pos="0.075 0.015 0" size="0.002" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# ==========================================
# 2. Bridge Platform XML
# ==========================================
platform_xml_content = """
<mujoco model="bridge_platform">
  <asset>
    <texture name="tex_platform" type="2d" builtin="checker" rgb1="0.1 0.1 0.1" rgb2="0.2 0.2 0.2" width="512" height="512"/>
    <material name="mat_platform" texture="tex_platform" shininess="0.1" specular="0.1" texrepeat="2 2"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom name="platform_geom" type="box" size="0.10 0.15 0.05" material="mat_platform" 
              density="5000" friction="1.0 0.005 0.0001" solref="0.001 1" solimp="0.95 0.99 0.001"
              condim="4" group="1"/>
        <geom name="platform_vis" type="box" size="0.10 0.15 0.05" material="mat_platform" 
              contype="0" conaffinity="0" group="0"/>
        <site name="top_site" pos="0 0 0.05" size="0.002" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# 写入逻辑 (保持路径正确)
brick_dir = os.path.join(ASSET_ROOT, "bridge_brick")
create_folder(brick_dir)
with open(os.path.join(brick_dir, "bridge_brick.xml"), "w") as f:
    f.write(brick_xml_content)

plat_dir = os.path.join(ASSET_ROOT, "bridge_platform")
create_folder(plat_dir)
with open(os.path.join(plat_dir, "bridge_platform.xml"), "w") as f:
    f.write(platform_xml_content)

print("\n✅ Assets regenerated (Soft Contact + Solref Fixed).")