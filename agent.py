# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:16:44 2018

@author: Leo
"""
#   历史战绩类以及操作
class Rules():
    def __init__(self, strategy_history):
        self. strategy_history = strategy_history
        
    #   更新历史战绩（删除第一个历史纪录，增加最近的战绩）
    def update_rules(self, rule):
        self.strategy_history.pop(0)
        self.strategy_history.append(rule)


#   确定主体的位置坐标
class Position():

    def __init__(self, position):
        self. x = position[0]
        self. y = position[1]


#    染色体:tag  control 片段（先只写标识tag片段）
class Chromosome():

    def __init__(self, offense_tag, defense_tag):
        self. offense_tag = offense_tag
        self. defense_tag = defense_tag
    
#   资源
class Resource():

    def __init__(self, founders, capitals, fans):
        self. founders = founders
        self. capitals = capitals
        self. fans = fans

#    主体
class Agent(Position, Chromosome, Rules):

    def __init__(self, position, 
                 offense_tag = None, 
                 defense_tag = None, 
                 founders = None, 
                 capitals = None, 
                 fans = None, 
                 strategy_history = None):
        #   坐标
        Position. __init__(self, position)
        #   染色体
        Chromosome. __init__(self, offense_tag, defense_tag)
        #   资源
        Resource. __init__(self, founders, capitals, fans)
        
        Rules.__init__(self, strategy_history)
        
        
    def get_position(self):
        return [self.x, self.y]
    
    def get_resource(self):
        return sum([self. founders, self. capitals, self. fans])
    
    
    
    
    
    
    
    
        
        








