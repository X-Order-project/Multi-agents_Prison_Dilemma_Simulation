# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 22:35:46 2018

@author: Leo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:29:41 2018

@author: Leo
"""
'''
此文档实现 主体数量和资源多少（或者agent适应度的大小）变化的可视化 
'''

import agent as gt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation


# 生成规则集(考虑前三次战绩)
def gene_all_set(n, rule_set):
    num = len(rule_set)
    rules = []
    temp = []    
    for i in range(num):
        for j in range(num):
            for k in range(num):  
                temp.append(rule_set[i])  
                temp.append(rule_set[j])
                temp.append(rule_set[k]) 
                rules.append(temp)
                temp = []
    return rules
    

#   生成基因序列(K:list,gene:list)
def gene_sequence(k, gene):
    genes = []
    gene_len = len(gene)
    for i in range(sum(k)):
        for j in range(gene_len): 
            if i < sum(k[:j+1]):
                genes.append(gene[j])
                break
    np.random.shuffle(genes)
    return genes

#   生成初始化战绩
def rand_start_rules(k, rule_basic):
    rule = []
    for i in  range(k):
        ind = np.random.randint(0,len(rule_basic))
        rule.append(rule_basic[ind])
    return rule

# 定义主体之间的距离
def distance_agents(agents):
    agents_num = len(agents)
    dists = []
    for i in range(agents_num):
        temp = []
        for j in range(agents_num):
            if i != j:
                d = np.sqrt((agents[j].y - agents[i].y)**2 + (agents[j].x - agents[i].x)**2)
            else:
                d = 0
            temp.append(d)
        dists.append(temp)
    return dists
   
#   两个主体交互，标识匹配    
def tag_match_score(offense_tag, defense_tag):
    c_num = 0
    len_o = len(offense_tag)
    len_d = len(defense_tag)
    len_match = min(len_o, len_d)
    score = 0
    for i in range(len_match):
        if offense_tag[i] == 'c' and defense_tag[i] =='c' :
            score += R
            c_num += 1
        elif offense_tag[i] == 'c' and defense_tag[i] =='d' :
            score += S
        elif offense_tag[i] == 'd' and defense_tag[i] =='c' :
            score += T
        elif offense_tag[i] == 'd' and defense_tag[i] =='d' :
            score += P
    
    score += (len_o - len_d)  
    return score, c_num

def tag_match_score1(offense_tag, defense_tag):
    c_num = 0
    score = 0

    if offense_tag == 'c' and defense_tag =='c' :
        score += R
        c_num += 1
    elif offense_tag == 'c' and defense_tag =='d' :
        score += S
    elif offense_tag == 'd' and defense_tag =='c' :
        score += T
    elif offense_tag == 'd' and defense_tag =='d' :
        score += P
    return score, c_num
    

#   单个主体在边界内随机移动
def brown_moven(agent):
    temp1 = np.random.randint(-1,2)
    temp2 = np.random.randint(-1,2)
   
    if agent.x + temp1 <= maxy and agent.x + temp1 >= maxx :
       agent.x += temp1
    if agent.y + temp2 <= maxy and agent.y + temp2 >= maxx:
       agent.y += temp2
    return agent


#   根据标识匹配得分（score）交换资源[+-,--,-+,++]
def exchange(agents, i, j):
    
    ind_i = all_rules.index(agents[i].strategy_history)
    ind_j = all_rules.index(agents[j].strategy_history)
    
    s1 = agents[i].offense_tag[ind_i]
    s2 = agents[j].offense_tag[ind_j]
    
    
    score_i, c1 = tag_match_score1(s1, s2)
    score_j, c2 = tag_match_score1(s2, s1)
        
    agents[i]. founders += score_i
    agents[j]. founders += score_j
    
    agents[i].update_rules(s1+s2)
    agents[j].update_rules(s2+s1)
    #   交互完成 两个主体随机游走
    brown_moven(agents[i])
    brown_moven(agents[j])  
#    print('Resource exchange has finished!')  
    return c1
    
            
 
#    获取所有主体适应度和，主体数量 
def get_agents_degree(agents):
    len_agents = len(agents)
    degree = 0
    for i in range(len_agents):
        degree += agents[i].get_resource()
    return degree, len_agents 
    
    
#   获取所有主体的 位置 和 资源信息
def get_agents_position(agents):
    position = []
    resource_total = []
    for agent in agents:
        position.append(agent.get_position())
        resource_total.append(agent.get_resource())
    return position, resource_total

# 增加新的主体
def generate_new_agent(agents, i, j):
    position = [agents[i].x, agents[i].y]
    new_agent = gt. Agent(position = position, 
                            offense_tag = agents[i]. offense_tag, defense_tag = agents[i].defense_tag, 
                            founders = agents[i].founders, capitals = agents[i]. capitals, fans = agents[i].fans,
                            strategy_history = agents[i].strategy_history)
    # 交换点
    change_point = np. random.randint(0, len(agents[i].defense_tag))
    # 交换
    new_agent.offense_tag = (new_agent.offense_tag)[:change_point]
    new_agent.offense_tag.extend((agents[j].offense_tag)[change_point:])
    new_agent.defense_tag = new_agent.offense_tag 
    
    # 适应度计算
    new_agent.founders = (new_agent.founders + agents[j].founders) / 2
#    new_agent.founders = 0

    # 突变
    new_agent = mutation_rule(new_agent)  
    new_agent = brown_moven(new_agent)
#    agents.append(new_agent)
    return new_agent

#   找出适应度最小的主体，满足一定比例的话删除
def del_min_degree(agents, len_new_agents):
    len_agents = len(agents)
    degree_all = []
    for i in range(len_agents):
        degree_all.append(agents[i].get_resource())
#    degree_min_rate = sum(degree_all) * del_rate
        
    #   删除增加的新的子代数量(保证群体中数量不变)
    for i in range(len_new_agents):
        ind = degree_all.index(min(degree_all))
        degree_all.pop(ind)
        agents.pop(ind)
        

    
    # 删除 小于 degree_min_rate 的所有Agent
#    for i in range(len_agents - 1, -1, -1):
#        if degree_all[i] <= degree_min_rate:
#            
##            agents.pop(i)
##            break
#            agents[i].founders = 0.001
#            
#            print('删除主体') 
#            break


def mutation_rule(agent):
    temp = np.random.randint(0,1000)
    if temp / 1000 < mutaion_rate:
        ind = np.random.randint(0, strategy_len)
        if agent.offense_tag[ind] == 'c':
            agent.offense_tag[ind] == 'd'
        else:
            agent.offense_tag[ind] == 'c'
        agent.offense_tag = agent.defense_tag
    return agent
    
    
#   主体之间进行进攻防御交互
def offense_defense(agents):

    dist = distance_agents(agents)
    # 加入如果适应度满足一定要求就进行产生子代 交换  删除适应度最低的那个
    degree_sum, len_agents = get_agents_degree(agents)   # 系统平均适应度   degree_sum / len_agents
    new_agents = []
    c_num_temp = []  # 每次博弈的合作对数目
    
    for i in range(len(agents)):
        temp = [k*(-1) for k in dist[i]]
        j = temp. index(min(temp))  # 与第 i 主体最近的主体 index = j (-号排除了离自己最近)
        print('平均适应度：',degree_sum / len_agents)
        
        #   两个主体适应度大于平均适应度就进行交互产生新的子代
        if (agents[i].get_resource()) >= degree_sum / (len_agents) and (agents[j].get_resource()) >= degree_sum / (len_agents) :
            print('交互并产生子代')
            # 满足繁殖条件进行产生子代（交换染色体片段）
            new_agent = generate_new_agent(agents, i, j)
            new_agents.append(new_agent)
            brown_moven(agents[i])
            brown_moven(agents[j])
            
            #   上面产生子代，下面交互 统计合作数目:c_num
            c_num = exchange(agents, i, j)
        else:
            print('交互')
            c_num = exchange(agents, i, j)

        c_num_temp.append(c_num)
    
    #  合作数目统计
    cooperation_num.append(sum(c_num_temp))
    
    del_min_degree(agents,len(new_agents))        
    agents.extend(new_agents)
    
#    print('len(agents):',len(agents))
    #   获取交互完成以后的主体 位置 和 资源
    position_agent, resource_total = get_agents_position(agents) 
    
    return position_agent, resource_total

#==========================================函数主体  上面定义小功能 ==============================



def data_process(maxx, maxy, agent_num, resource_kind, resource_num):
    #   生成策略集
#    all_rules = gene_all_set(3, ['cc','cd','dc','dd'])
    
    
    # 路由矩阵
    agents_mat = np.zeros([maxy - maxx, maxy - maxx])   # 主体路由矩阵
#    resource_mat = np.zeros([maxx, maxy])   # 资源路由矩阵(暂时无用)
    
    
    # ==================================主体定义==================================================
    # 随机 主体定义起始点位置
    position_agent = np.random.randint(maxx, maxy, size=(agent_num, 2))  
    
    agents = []
    temp = []
#    strategy_len = 16   # 策略长度
    for i in range(agent_num):
        # 标识初始化
        temp1 = np.random.randint(0, strategy_len)
        temp = list([temp1])
        temp.append(strategy_len - temp1)
        
        offense = gene_sequence(temp, ['c','d'])
        defense = offense
        
        # 主体资源初始化        
        founders = 0
        capitals = 0.0001
        fans = 0
        
        #   历史策略初始化
        strategy_history = rand_start_rules(3,['cc','cd','dc','dd'])
                    
        agents.append(gt. Agent(position = position_agent[i], 
                                offense_tag = offense, 
                                defense_tag = defense, 
                                founders = founders, 
                                capitals = capitals, 
                                fans = fans,
                                strategy_history = strategy_history))
        
        agents_mat[position_agent[i][0]][position_agent[i][1]] = 1   # 标记有此位置有一个主体
      
        
        
    #=========================世界位置   资源初始化 【暂时不考虑】======================================================================
    resources = []
    
    for i in range(maxx):
        res_row = []
        for j in range(maxy):
            temp = list(np.random.randint(0, 5, size=(1, 3))[0])  # 分别代表三种资源的数量
            res_row.append(temp) 
        resources.append(res_row)
    return agents, agents_mat
            
     
# ===================================== 可视化过程 =====================================        
def init():
#    p1.set_offsets([])
    p1.set_data([], [])
    p2.set_data([], [])
    return p1,p2 

def update_scatter(i):
    #    获取新的数据
    
#    for i in range(1):
    position_agent, resource_total = offense_defense(agents) 
#    x = np.array(position_agent)[:,0]
#    y = np.array(position_agent)[:,1]
#    print(resource_total)    
    
    #   计算每个周期平均收益率
    temp = earn_avg[-1]
    earn_avg.pop(-1)
    earn_avg.append((sum(resource_total) - temp) / len(resource_total))
    earn_avg.append(sum(resource_total))

    
    x = np.array(list(range(len(earn_avg)-1)))
    y = np.array(earn_avg[:-1]) 
        
    p1.set_data(x, y)
    ax1.set_xlim(0, len(x) + 10)
    ax1.set_ylim(min(y)-10, max(y) + 10) 
#    ax1.set_xlabel('frame {0}'.format(i))
    
    
    # 每个周期合作次数
    p2.set_data(list(range(len(cooperation_num))), cooperation_num)
    ax2.set_xlim(0, len(cooperation_num) + 10)
    ax2.set_ylim(0, max(cooperation_num) + 10)   
    
    
    #    传入新的数据
#    p1.set_offsets([x,y])
#    p1._sizes = np.array(resource_total)   # Set sizes...
#    p1.set_array(np.array(resource_total) / max(np.array(resource_total)) )   # Set colors..
#    ax1.set_xlabel('frame {0}'.format(i))
    
    
    #    传入新的数据  
#    degree_sum, len_agents = get_agents_degree(agents)  
#    degree_avg.append( degree_sum / len_agents) 
#    print('degree_avg: ', degree_avg)
    
#    xx = list(range(len(degree_avg)))
#    print(xx)
#    p2.set_data(xx, degree_avg)
#    p2.set_offsets([x,y])
#    p2.set_offsets([xx, degree_avg])
    
#    ax2.set_xlim(0, len(degree_avg) + 10)
#    ax2.set_ylim(0, max(degree_avg) + 10)    
       
    return p1,p2


if __name__ == '__main__':
    
    # 数据预处理，定义位置、主体、资源  （预设）
    global agents
    global maxx, maxy
    global R, S, T, P
    global del_rate
    global strategy_len
    global all_rules
#    global degree       #平均总体适应度
    
#   统计特性：
    global degree_avg
    global earn_avg
    global cooperation_num
    
    
    global mutaion_rate
    
    
    mutaion_rate = 0.005  # 突变率
    
    
    degree_avg = [0] # 平均适应度
    earn_avg = [0,0]  #  每次博弈平均收益
    cooperation_num = [0] # 合作次数（C，C）
    
 
    R, S, T, P = 3, 5, 0, 1   
    maxx, maxy = 0,100   # 世界的大小
    agent_num = 20         # 主体的起始数量
    resource_kind = 3       # 资源种类
    resource_num = 20      # 资源数量
    del_rate = 0.2        # 删除比例
    strategy_len = 64   # 策略长度
    np.random.seed( 10 )
    all_rules = gene_all_set(3, ['cc','cd','dc','dd'])
    
    #======================= 预处理（以后为外部数据输入窗口）==================================  
    agents, agents_mat = data_process(maxx, maxy, agent_num, resource_kind, resource_num)
        
    #===============================主体交互作用 ==============================================
    #   获得主体的位置以及大小（根据资源大小确定）
    position_agent, resource_total = get_agents_position(agents)    
#    x = np.array(position_agent)[:,0]
#    y = np.array(position_agent)[:,1] 
    
    # ============================= 可视化 ===================================================
#    fig = plt.figure(figsize=(10, 10), facecolor="white")
#    ax = fig.add_subplot(111)
#    p1, = ax.plot(x, y, 'o', lw = 2)       
    
    fig = plt.figure(figsize = (10, 10))
#    ax1 = plt.axes(xlim=(maxx - 20, maxy + 20), ylim=(maxx - 20, maxy + 20))    
#    ax1 = fig.add_subplot(1,1,1, xlim=(0, 2), ylim=(0, 2))
    ax1 = fig.add_subplot(2,1,1)
    plt.grid(True)
    plt.title("The overall average earning ")
    plt.xlabel('Time')
    plt.ylabel('Average Earning')
    p1, = ax1.plot([], [], lw=1)
    
    
    
#   第二个指标构建        
    ax2 = fig.add_subplot(2,1,2, xlim=(0, 200), ylim=(0, 200))
           
#    plt.colorbar(p1)
    plt.grid(True)
    plt.title("The number of cooperation ")
    plt.xlabel('Time')
    plt.ylabel('Cooperation number')
    
##    ax2 = plt.axes(xlim=(maxx - 20, maxy + 20), ylim=(maxx - 20, maxy + 20))
    p2, = ax2.plot(list(range(len(cooperation_num))), cooperation_num, lw=1)
    



    anim = FuncAnimation(fig, update_scatter, init_func=init, frames = range(1000), interval = 5, repeat = False)
#    anim = FuncAnimation(fig, update_scatter, frames = range(200), blit=True, interval=500)
    plt.show() 
    
    
    
    
    
    
    
#    anim.save('/tmp/CopyAndDeath.gif', writer='imagemagick', fps=30)
    
    mywriter = animation.FFMpegWriter(fps=10)
    anim.save('D:/xtoken/complex_system/pd/result/Cf2.mp4',writer=mywriter) 
#    
#    anim.save('line.gif', dpi=80, writer='imagemagick')
    
    
#    anim.save('/result/r.mp4')  
    
#    from IPython.display import HTML
#    HTML(anim.to_html5_video())    


# 交互作用



# 可视化



