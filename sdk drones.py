from robomaster import robot
import time
if __name__ == '__main__':
    tl_drone = robot.Drone()    #创建实例类对象
    tl_drone.initialize()   #初始化
    tl_camera=tl_drone.camera
    # 切换飞行器WiFi模式为组网模式，指定路由器SSID和密码（仅此法可连）
    tl_drone.config_sta(ssid="RMTT-55F5D8", password="12341234")
    tl_flight = tl_drone.flight
    tl_flight.takeoff().wait_for_completed()    #wait for completed()安全
    tl_flight.up(distance=50).wait_for_completed()  # 向上飞50cm
    tl_flight.flip_forward().wait_for_completed()
    tl_flight.flip_backward().wait_for_completed()
    #tl_flight.forward(distance=50).wait_for_completed() #向前飞50cm
    time.sleep(10)
    tl_flight.land().wait_for_completed()   #降落
    tl_drone.close()