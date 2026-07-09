Chinese character involved in this project, mind encoding!
使用myml python环境,位于D:\Software\anaconda\envs\myml. torch版本为2.9.0+cu128
工作目录中是fork的别人的生成递归编码器框架。我需要你帮我完善代码并扩充，以用于飞行器布局的生成
Grass-matlab中是作者的原始matlab实现，禁止修改，不要commit，鼓励查询并参考
mistakes部分中记录了过去常犯的错误，必须阅读保证不再犯错。如果有新的frequent mistake同样记录在mistakes部分中

开发规则：
1. 如无必要禁止修改代码原有结构。修改结构需给出清晰依据

2. DRY,拒绝知识的重复。系统中的每一个功能点、算法或配置，都应有且仅有一个权威定义。禁止在多个地方手动同步相同的逻辑.

平衡点： 避免为了 DRY 而引入过度复杂的泛型或多层继承。如果消除重复会导致代码可读性急剧下降，请优先选择代码的清晰度，并辅助以显式注释.

3. 单一来源,常量、魔术字符串, 数据库 Schema 必须定义在集中配置文件中.

4. Fail-Fast 机制暴露错误
禁止过度防御性编程，不使用 config.get('max_workers', 4)的默认参数，必须让潜在的错误直接通过报错暴露出来


mistakes:
(无)
