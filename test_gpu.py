import mujoco
import glfw

# 初始化 GLFW
if not glfw.init():
    print("❌ 错误: 无法初始化 GLFW (这通常意味着缺少显示设备或驱动问题)")
else:
    # 创建一个隐藏窗口来测试 OpenGL 上下文
    window = glfw.create_window(640, 480, "Test Window", None, None)
    if not window:
        print("❌ 错误: 无法创建 GLFW 窗口")
        glfw.terminate()
    else:
        print("✅ 成功: 你的 Ubuntu 系统可以正常弹出交互窗口！")
        glfw.destroy_window(window)
        glfw.terminate()
