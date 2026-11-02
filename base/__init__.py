import os           # 用于处理文件路径
import sys          # 用于管理Python解释器的系统路径，确保自定义模块能被正确导入.

# 1.获取当前文件所在目录的绝对路径，即:当前脚本所在的文件夹.
module_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'module_dir: {module_dir}')         # E:\LLM_ubuntu\integrated_qa_system\base

# 2.获取项目根目录的绝对路径.
# 用途:将项目根目录加入系统路径，确保跨目录导入自定义模块(例如:config,Logger)
project_root = os.path.dirname(module_dir)
# print(f'project_root: {project_root}')
"""
“工程级”的代码。
未来会用什么工具运行，不知道你会不会把这个项目部署到服务器上
为了防止你在终端、服务器、其他电脑上运行时报错
如果你不用 PyCharm，改用命令行（终端）直接运行你那个深层的脚本，就会瞬间报错,所以写到系统路径前面，一下就能找到
"""
# 3.将模块目录加入系统路径.
# 解释:若模块目录不再sys.path中，插入到列表首位.确保优先加载当前模块目录下的文件，避免:同名模块冲突
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

# 4.将项目根目录加入系统路径.
# 解释:如果项目根目录不再sys.path中，插入到列表首位，确保优先加载项目根目录下的文件，避免:同名模块冲突
if project_root not in sys.path:
    sys.path.insert( 0, project_root)

# 5.扩展:调用并打印.
# print(sys.path)

# 6.导入自定义模块.
"""
在 Python 中，__init__.py 文件的一个核心作用就是把包里面零散的功能，打包成一个“统一的对外出口”。
如果 __init__.py 里写了这两行,那么当其他代码（比如项目外面的脚本，或者同项目的其他模块）导入这个包时，可以直接这样写：
from base import Config, logger   # ✅ 直接拿到，非常简洁
但如果第 6 点被注释掉了，别人就只能深入包内部去导入：
from base.config import Config    # 必须写到具体文件，很繁琐
from base.logger import logger
"""
from config import Config       # 导入配置解析类，用于:读取项目配置.
from logger import logger       # 导入初始化好的日志实例，用于项目日志记录。

