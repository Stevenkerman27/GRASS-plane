Chinese character involved in this project, mind encoding!
使用myml python环境,位于D:\Software\anaconda\envs\myml. torch版本为2.9.0+cu128
工作目录中是fork的别人的GRASS框架。我需要你帮我完善代码并扩充
使用context7 MCP查询pytorch文档
理解任务后先与用户讨论想法, must ensure alignment before start editing
mistakes部分中记录了过去常犯的错误，必须阅读保证不再犯错。如果有新的frequent mistake同样记录在mistakes部分中

开发规则：
1. 如无必要禁止修改代码原有结构。修改结构需给出清晰依据

2. DRY,拒绝知识的重复。系统中的每一个功能点、算法或配置，都应有且仅有一个权威定义。禁止在多个地方手动同步相同的逻辑.

平衡点： 避免为了 DRY 而引入过度复杂的泛型或多层继承。如果消除重复会导致代码可读性急剧下降，请优先选择代码的清晰度，并辅助以显式注释.

3. 单一来源,常量、魔术字符串, 数据库 Schema 必须定义在集中配置文件中.

4. 在任务完成后，梳理本次开发产生的架构变动、新的防御性编程基线等，将它们提炼为全局代码层面的客观规律，更新到全局 progress 文档的对应模块中。删除已修复的短期 Bug 细节以及过时的模块描述，保持该文档作为‘当前系统架构与开发约束准则’的纯粹性。不要新增progress条目

mistakes:
1. 当在CLI执行 Python 代码块时，必须遵循以下原则：不要在 f-string 中使用引号嵌套；禁止在 {} 内出现反斜杠；严格引号分层。禁止在 f-string 的大括号内使用任何引号或反斜杠。如果需要打印字典内容，使用 print('text', dict['key']) 这种多参数形式，不要在字符串内部嵌套转义引号

2. 不要尝试执行单行复杂命令，换行语法容易出错。2行以上命令必须写出临时代码再执行