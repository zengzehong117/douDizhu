# douDizhu

基于流行游戏“斗地主”，使用深度强化学习算法，开发的游戏Ai。（规则有所变化，游戏人数变为两人，双方明牌，没有王，及其他）

实现了训练、模拟环境（是否通用）、神经网络输入设计、神经网络设计、激活函数设计、损失函数设计、动作空间等等。

算法达到想要效果，目前是初级版本。

算法创新点是：复杂动作空间设计。

测试示例：


init player Ai: ('3', '4', '5', '5', '6', '6', '8', '8', '8', '9', 'J', 'K', 'A', 'A', '2')

init player random: ('4', '5', '7', '9', '10', '10', 'J', 'J', 'Q', 'Q', 'Q', 'K', 'K', '2', '2')

player Ai  put card: ('4', '8', '8', '8')

player Random put card: ('4', 'Q', 'Q', 'Q')

player Ai:  过.

player Random put card: J

player Ai  put card: 2

player Random:  过.

player Ai  put card: ('5', '5')

player Random put card: ('2', '2')

player Ai:  过.

player Random put card: J

player Ai  put card: K

player Random:  过.

player Ai  put card: ('6', '6')

player Random put card: ('10', '10')

player Ai  put card: ('A', 'A')

player Random:  过.

player Ai  put card: 3

player Random put card: 5

player Ai  put card: 9

player Random put card: K

player Ai:  过.

player Random put card: 7

player Ai  put card: J

Winer is player Ai.

