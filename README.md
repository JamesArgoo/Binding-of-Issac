# Binding-of-Issac
An AI to play Issac，不采取图像识别的方法，而是去提取游戏运行内存中的数据。文件夹isaac_ai_bridge是一个mod，放置到你的游戏目录（比如D:\SteamLibrary\steamapps\common\The Binding of Isaac Rebirth\mods）里面，steam中Isaac游戏设置-属性-高级选项里面填上“--luadebug”，用于启动这个提取游戏数据的mod.打开游戏金额isaac_bridge.py如果能够看到输出说明能够连接到游戏。



ai_basic.py是一个基于规则的AI，目前还算可以。

bc_data_human.npz是人类玩家游玩的数据记录，产生于record_human.py

bc_train用来把人类玩家的游玩数据训练给AI，这个效果不佳

isaac_bridge.py  diag.py仅用于测试，测试是否能够获取数据。

两个有帮助的以撒的结合接口网址：https://cuerzor.github.io/IsaacDocs/rep/Options.html，https://moddingofisaac.com/docs/rep/enums/GridEntityType.html