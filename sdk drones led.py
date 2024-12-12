import time
import robomaster
import socketpip
from robomaster import robot

if __name__ == '__main__':
# 设置ip地址和端口（默认）
      tello_ip = "192.168.10.1"
      tello_port = 8889

      tl_drone = robot.Drone()
# 初始化
      tl_drone.initialize()
      tl_led=tl_drone.led
      tl_drone.config_sta(ssid="RMTT-55F5D8", password="12341234")
      tl_led.set_led(r=0,g=0,b=0)
# 创建UDP socket,TCP连接
      sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      sock.bind(("", 9000))
# 进入SDK模式
      sock.sendto(b"command", (tello_ip, tello_port))
      time.sleep(1)
# I
      sock.sendto(b"EXT mled g 00rrr000000r0000000r0000000r0000000r0000000r0000000r000000rrr000", (tello_ip, tello_port))
      time.sleep(2)
# love
      sock.sendto(b"EXT mled g 0rr00rr00rrrrrr0rrrrrrrrrrrrrrrr0rrrrrr000rrrr00000rrr00000r0000", (tello_ip, tello_port))
      time.sleep(2)
# U
      sock.sendto(b"EXT mled g 0r0000r00r0000r00r0000r00r0000r00r0000r00r0000r00r0000r00rrrrrr0", (tello_ip, tello_port))
      time.sleep(2)
# 关闭socket
      sock.close()
      tl_drone.close()
