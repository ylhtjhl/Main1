import time
from itertools import permutations
import heapq
import numpy as np
import RPi.GPIO as GPIO
import time
import YB_Pcb_Car  # 导入Yahboom专门库文件
import threading
import HSV
from path_CES import Treasure

car = YB_Pcb_Car.YB_Pcb_Car()
Tracking_Left1 = 13  # X1B 左边第一个传感器
Tracking_Left2 = 15  # X2B 左边第二个传感器
Tracking_Right1 = 11  # X1A  右边第一个传感器
Tracking_Right2 = 7  # X2A  右边第二个传感器

GPIO.setmode(GPIO.BOARD)

GPIO.setwarnings(False)

GPIO.setup(Tracking_Left1,GPIO.IN)
GPIO.setup(Tracking_Left2,GPIO.IN)
GPIO.setup(Tracking_Right1,GPIO.IN)
GPIO.setup(Tracking_Right2,GPIO.IN)



#最基本的A*搜索算法
def A_star(map, start, end):
    """
    使用A*算法寻找起点到终点的最短路径
    :param map: 二维列表，表示地图。0表示可以通过的点，1表示障碍物。
    :param start: 元组，表示起点坐标。
    :param end: 元组，表示终点坐标。
    :return: 列表，表示从起点到终点的最短路径，其中每个元素是一个坐标元组。
    """
    # 定义启发式函数（曼哈顿距离）
    def heuristic(node1, node2):
        return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

    # 初始化open_list、closed_list、g_score、came_from
    open_list = [(0, start)]
    closed_list = set()
    g_score = {start: 0}
    came_from = {}

    # 开始搜索
    while open_list:
        # 取出f值最小的节点
        current = heapq.heappop(open_list)[1]
        if current == end:
            # 找到终点，返回路径
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        # 将当前节点加入closed_list
        closed_list.add(current)

        # 遍历相邻节点
        for neighbor in [(current[0] - 1, current[1]),
                         (current[0] + 1, current[1]),
                         (current[0], current[1] - 1),
                         (current[0], current[1] + 1)]:
            if 0 <= int(neighbor[0]) < len(map) and 0 <= int(neighbor[1]) < len(map[0]) and map[int(neighbor[0])][int(neighbor[1])] == 0:
                # 相邻节点是可通过的节点
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # 如果相邻节点不在g_score中，或者新的g值更优，则更新g_score和came_from
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_list, (f_score, neighbor))
                    came_from[neighbor] = current

    # 没有找到可行路径，返回空列表
    return []

#坐标变换函数，将10*10的坐标映射到地图矩阵上，方便用来可视化
def pose2map(x,y):
    return 21-2*y,x*2-1

def map2pose(x,y):
    return (21-y)/2,(x+1)/2

#地图上两点之间的最短路径
def A_star_length(map1,start_x,start_y,end_x,end_y):
    # 定义起点和终点
    start = (start_x, start_y)
    end = (end_x, end_y)

    # 计算最短路径
    start=pose2map(*start)
    end=pose2map(*end)
    path = A_star(map1, start, end)
    path_length=int((len(path)-1)/2)
    return path_length

#预计算
def precomputation(map1,start,end,mid_points):
    permutations_list = list(permutations(mid_points,2))
    length_dict={}
    length_dict[start]={}
    for pt1 in mid_points:
        length_dict[pt1]={}
        length_dict[start][pt1]=A_star_length(map1,start[0],start[1],pt1[0],pt1[1])
    for pt1 in mid_points:
        length_dict[pt1][end]=A_star_length(map1,pt1[0],pt1[1],end[0],end[1])
    for pt1,pt2 in permutations_list:
        length_dict[pt1][pt2]=A_star_length(map1,pt1[0],pt1[1],pt2[0],pt2[1])
    length_dict[start][end]=A_star_length(map1,start[0],start[1],end[0],end[1])
    return length_dict

#计算最短距离的路线
def get_min_path(map1,start,end,mid_points,length_dict=None):
    #穷举法8！=40320
    #计算1000个路径需要3s，全部计算需要2分钟计算太慢,但是使用路径查询后大大减少了计算量40320组数据在0.2s完成计算获得最优路径
    permutations_list = list(permutations(mid_points))
    min_path_length=float("inf")
    min_path=None
    for mid_points in permutations_list:
        mid_points=list(mid_points)
        mid_points.append(end)
        mid_points.insert(0,start)

        all_length=0
        for i in range(len(mid_points)-1):
            if length_dict:#如果没有预计算则采用现场计算，很费时
                length=length_dict[mid_points[i]][mid_points[i+1]]
            else:
                length=A_star_length(map1,mid_points[i][0],mid_points[i][1],mid_points[i+1][0],mid_points[i+1][1])
            all_length+=length
        if all_length<min_path_length:
            min_path_length=all_length
            min_path=mid_points

    return min_path,min_path_length 

#将10*10pose坐标映射到21*21的地图坐标上
def gennerate_all_path(map1,min_path):
    path=[]
    for i in range(len(min_path)-1):
        #start=pose2map(*start)
        #end=pose2map(*end)
        base_path=A_star(map1,pose2map(*min_path[i]),pose2map(*min_path[i+1]))
        path+=base_path[1:]
    path.insert(0,pose2map(*min_path[0]))
    return path

def multi_goal_Astar(map1,start,end,mid_points):
    '''
    含有中间位置的最短路径规划算法
    '''
    yujisuan=precomputation(map1,start,end,mid_points)
    min_path,min_path_length =get_min_path(map1,start,end,mid_points,yujisuan)

    #print(real) 
    
    return min_path,min_path_length

def multi_Astar(map1,start,end,mid_points):
    min_path,min_path_length=multi_goal_Astar(map1,start,end,mid_points)
    all_points=[]
    for i in range(len(min_path)-1):
        temp=A_star(map1,pose2map(*min_path[i]),pose2map(*min_path[i+1]))[:-1]
        for j in temp:
            all_points.append(j)
    all_points.append(pose2map(*min_path[-1]))
    real=[]
    for point in all_points:
        real.append(map2pose(point[0],point[1]))
    #print(all_points)
    real=real[::-1]
    return real
def find_turning_points(path):
    turning_points = []
    for i in range(1, len(path)-1):
        current = path[i]
        previous = path[i-1]
        next = path[i+1]
        if ((current[0]-previous[0]) * (next[1]-current[1]) != (current[1]-previous[1]) * (next[0]-current[0])) or(current[0]-previous[0]) * (next[0]-current[0]) + (current[1]-previous[1]) * (next[1]-current[1]) <0:#or (current[0]-previous[0]) * (next[1]-current[1])==(current[1]-previous[1]) * (next[0]-current[0]) ==0#(3,1) (3,3) (3,1) (3-3)*(1-3) (3-1)*(1-3)
            turning_points.append(current)


    return turning_points

def turn_direction(v1, pt1,pt2):
    x1,y1=pt1
    x2,y2=pt2
    # 计算下一个方向向量
    v2 = np.array([x2, y2]) - np.array([x1, y1])
    # 计算叉积
    cross = np.cross(v1, v2)
    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
    norm_product = ((v1[0]**2 + v1[1]**2) *(v2[0]**2 + v2[1]**2))**0.5
    if cross > 0:
        return "left"
    elif cross < 0:
        return "right"
    elif dot_product==norm_product:
        return "straight"
    else:
        return "Reverse_direction"

def car_move(action):
    if action == "right":
        # 右转，左边的轮子正转，右边的轮子反转
        while 1:
            car.Car_Spin_Right(100, 100)
            time.sleep(0.05)
            Tracking_Left1Value = GPIO.input(Tracking_Left1)
            Tracking_Left2Value = GPIO.input(Tracking_Left2)
            Tracking_Right1Value = GPIO.input(Tracking_Right1)
            Tracking_Right2Value = GPIO.input(Tracking_Right2)
            if Tracking_Left2Value == 1 and Tracking_Right1Value == 1:
                break
        car.Car_Stop()
        
    elif action == "left":
        # 左转，左边的轮子反转，右边的轮子正转
        while 1:
            car.Car_Spin_Left(70, 70)
            
            Tracking_Left1Value = GPIO.input(Tracking_Left1)
            Tracking_Left2Value = GPIO.input(Tracking_Left2)
            Tracking_Right1Value = GPIO.input(Tracking_Right1)
            Tracking_Right2Value = GPIO.input(Tracking_Right2)
            if Tracking_Left2Value == 1 and Tracking_Right1Value == 1:
                break
        car.Car_Stop()
        
    elif action == "straight":
        while 1:
            Tracking_Left1Value = GPIO.input(Tracking_Left1);
            Tracking_Left2Value = GPIO.input(Tracking_Left2);
            Tracking_Right1Value = GPIO.input(Tracking_Right1);
            Tracking_Right2Value = GPIO.input(Tracking_Right2);
            if Tracking_Left1Value == True and Tracking_Left2Value == False and Tracking_Right1Value == False and Tracking_Right2Value == True:
                car.Car_Run(70, 70)
            
            
               # 四路循迹引脚电平状态
                # 1 0 1 1
                # 往右微调
            elif Tracking_Left1Value == True and Tracking_Left2Value == False and Tracking_Right1Value == True and Tracking_Right2Value == True:
                car.Car_Run(50, 70)
                car.Car_Spin_Left(70, 70)
                time.sleep(0.02)
                # 四路循迹引脚电平状态
                # 1 1 0 1
                # 往左微调
            elif Tracking_Left1Value == True and Tracking_Left2Value == True and Tracking_Right1Value == False and Tracking_Right2Value == True:
                car.Car_Spin_Right(70, 70)
                time.sleep(0.02)
            else: 
                return 0
        car.Car_Stop()
    
    
    elif action == "Reverse_direction":
        # 掉头，两边的轮子都反转
        while 1:
            car.Car_Spin_Right(100, 100)
            time.sleep(0.05)
            Tracking_Left1Value = GPIO.input(Tracking_Left1)
            Tracking_Left2Value = GPIO.input(Tracking_Left2)
            Tracking_Right1Value = GPIO.input(Tracking_Right1)
            Tracking_Right2Value = GPIO.input(Tracking_Right2)
            if Tracking_Left2Value == 1 and Tracking_Right1Value == 1:
                break
        car.Car_Stop()
        

    elif action =="Back":
        car.Car_Back(70, 70)
        time.sleep(0.5)
    

maze_color=0
lukou=0
count=0
def move_with_one_line():
    
    while True:
        global maze_color
        global lukou
        global count
        tracking_function()
        #time.sleep(0.5)
        count=count+1
        if lukou==1:#识别到岔路口跳出巡线
            print("识别到路口")
            return 0


        if(count==25):#每25次循环调用一次宝藏识别函数
            print("进行一次宝藏识别")
            maze_color = HSV.colors()#调用宝藏识别函数给color赋值
            count=0 #计数值清零

        
        if maze_color==0:#无宝藏
            pass
        if maze_color==1:#己方宝藏
            pass
        if maze_color==2: #对方宝藏
            pass
    return maze_color

def tracking_function():
    Tracking_Left1Value = GPIO.input(Tracking_Left1);
    Tracking_Left2Value = GPIO.input(Tracking_Left2);
    Tracking_Right1Value = GPIO.input(Tracking_Right1);
    Tracking_Right2Value = GPIO.input(Tracking_Right2);
    global lukou
    global color
        
       # 四路循迹引脚电平状态
        # 1 0 0 1
        # 处理直线
    if Tracking_Left1Value == True and Tracking_Left2Value == False and Tracking_Right1Value == False and Tracking_Right2Value == True:
        car.Car_Run(70, 70)

            # 四路循迹引脚电平状态
            # 1 0 1 1
            # 往右微调
        if Tracking_Left1Value == True and Tracking_Left2Value == False and Tracking_Right1Value == True and Tracking_Right2Value == True:
            car.Car_Spin_Left(70, 70)
            time.sleep(0.02)
            # 四路循迹引脚电平状态
            # 1 1 0 1
            # 往左微调
        elif Tracking_Left1Value == True and Tracking_Left2Value == True and Tracking_Right1Value == False and Tracking_Right2Value == True:
            car.Car_Spin_Right(70, 70)
            time.sleep(0.02)


        # 当为1 1 1 1时小车保持上一个小车运行状态
    elif Tracking_Left1Value == False or Tracking_Right2Value == False :
        car.Car_Stop()
        lukou=1


def attactk():
    """attactk 撞击宝藏
    """
    car_move("Reverse_direction")#执行掉头动作
    car_move("Back")
    car_move("straight")
    
   
    return

if __name__=="__main__":

    maze_location=[]
    
    start=(1, 1)
    end=(10,10)

    map1=[
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1],
[1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1],
[1,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1],
[1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1],
[1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1],
[1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1],
[1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1],
[1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1],
[1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1],
[1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1],
[1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1],
[1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1],
[1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1],
[1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1],
[1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1],
[1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1],
[1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1],
[1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]
    print("开始路径规划")
    min_path=multi_Astar(map1,start,end,maze_location)
    turn_points=find_turning_points(min_path)
    #turn_points.insert(0,(1,1))
    #turn_points=min_path
    print("路径规划已完成")
    print(turn_points)
    now_dir=(1,0)
    print("设定初始方向",now_dir)
    cross_points=[(3 ,4),
(4 ,3),
(5 ,2),
(6 ,2),
(7 ,1),
(9 ,1),
(8 ,3),
(7 ,4),
(7 ,5),
(7 ,6),
(8 ,6),
(10, 6),
(8 ,7),
(7 ,8),
(6 ,9),
(5 ,9),
(3 ,8),
(4 ,7),
(4 ,6),
(3 ,5),
(1 ,5),
(4 ,5),
(4 ,10),
(2 ,10)]
    print("准备动作执行列表")
    global action_list
    action_list=[]
    for i in range(len(turn_points)-1):
        
        action=turn_direction(now_dir, turn_points[i], turn_points[i+1])
        now_dir=turn_points[i+1][0]-turn_points[i][0],turn_points[i+1][1]-turn_points[i][1]
        print(i,turn_points[i],action)
        #if turn_points[i] in cross_points:# or action=="Reverse direction":
            #print(i,turn_points[i],action)
        action_list.append(action)
    #遇到多叉路口执行动作的列表"right" "left" "straight" "Reverse direction"
    #开始寻线行驶
    print("开始寻线行驶")
    move_with_one_line()
    #先把小车头朝迷宫放置。

    for action in action_list:
    #action = action_list.pop(0)
        print("到达岔路口选择动作",action)
        print(action_list)
        car_move(action)#多叉口路口选择方向
        color=move_with_one_line()#寻线行驶返回条件有两个 1到达新的岔路口 2识别到宝藏

        if maze_color==0:#没有识别到宝藏
            continue

        elif maze_color==1:#己方宝藏
            pass
            attactk()#碰撞动作
            car_move("Reverse_direction")#执行掉头动作

        else:#对方宝藏
            car_move("Reverse_direction")#执行掉头动作
            pass
        
    #到达终点
    print("到达终点结束运行")
