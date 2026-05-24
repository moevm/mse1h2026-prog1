from .submodule_1.task_1 import StructDeclTask, StructDeclCLIParser
from .submodule_1.task_2 import StructInitTask, StructInitCLIParser
from .submodule_1.task_3 import StructComputeTask, StructComputeCLIParser
from .submodule_1.task_4 import StructAlignTask, StructAlignCLIParser
from .submodule_1.task_5 import StructPaddingTask, StructPaddingCLIParser
from .submodule_1.task_6 import StructOptimalTask, StructOptimalCLIParser
from .submodule_1.task_7 import StructSizeofTask, StructSizeofCLIParser

from .submodule_2.task_1 import UnionFillTask, UnionFillCLIParser
from .submodule_2.task_2 import UnionBytesTask, UnionBytesCLIParser
from .submodule_2.task_3 import UnionSizesTask, UnionSizesCLIParser

from .submodule_3.task_1 import EnumNextTask, EnumNextCLIParser
from .submodule_3.task_2 import EnumCombineTask, EnumCombineCLIParser

from .submodule_4.task_1.task_1_1 import TypedefAliasTask, TypedefAliasCLIParser
from .submodule_4.task_1.task_1_2 import TypedefPrintTask, TypedefPrintCLIParser
from .submodule_4.task_2 import TypedefStructTask, TypedefStructCLIParser

from .submodule_5.task_1 import FilePointerTask, FilePointerCLIParser
from .submodule_5.task_2 import FOpenTask, FOpenCLIParser
from .submodule_5.task_3 import FCloseTask, FCloseCLIParser
from .submodule_5.task_4 import FReadTask, FReadCLIParser
from .submodule_5.task_5 import FWriteTask, FWriteCLIParser
from .submodule_5.task_6.task_6_1 import TextIOTask, TextIOCLIParser
from .submodule_5.task_6.task_6_2 import BinaryIOTask, BinaryIOCLIParser
from .submodule_5.task_7 import QsortTask, QsortCLIParser
from .submodule_5.task_8 import BsearchTask, BsearchCLIParser
from .submodule_5.task_9.task_9_1 import ExitTask, ExitCLIParser
from .submodule_5.task_9.task_9_2 import ExitFailureTask, ExitFailureCLIParser
from .submodule_5.task_10 import AtexitTask, AtexitCLIParser
from .submodule_5.task_11 import GetEnvTask, GetEnvCLIParser
from .submodule_5.task_12 import SystemTask, SystemCLIParser

from .submodule_6.task_1 import ErrnoTask, ErrnoCLIParser
from .submodule_6.task_2 import PerrorTask, PerrorCLIParser
from .submodule_6.task_3 import ErrorHandlingTask, ErrorHandlingCLIParser

from .submodule_7.task_1 import NestedStructTask, NestedStructCLIParser
from .submodule_7.task_2 import FlexibleArrayTask, FlexibleArrayCLIParser
from .submodule_7.task_3 import BitFieldsTask, BitFieldsCLIParser

__all__ = [
    "StructDeclTask",
    "StructDeclCLIParser",
    "StructInitTask",
    "StructInitCLIParser",
    "StructComputeTask",
    "StructComputeCLIParser",
    "StructAlignTask",
    "StructAlignCLIParser",
    "StructPaddingTask",
    "StructPaddingCLIParser",
    "StructOptimalTask",
    "StructOptimalCLIParser",
    "StructSizeofTask",
    "StructSizeofCLIParser",

    "UnionFillTask",
    "UnionFillCLIParser",
    "UnionBytesTask",
    "UnionBytesCLIParser",
    "UnionSizesTask",
    "UnionSizesCLIParser",

    "EnumNextTask",
    "EnumNextCLIParser",
    "EnumCombineTask",
    "EnumCombineCLIParser",

    
    "TypedefAliasTask",
    "TypedefAliasCLIParser",
    "TypedefPrintTask",
    "TypedefPrintCLIParser",
    "TypedefStructTask",
    "TypedefStructCLIParser",

    "FilePointerTask",
    "FilePointerCLIParser",
    "FOpenTask",
    "FOpenCLIParser",
    "FCloseTask",
    "FCloseCLIParser",
    "FReadTask",
    "FReadCLIParser",
    "FWriteTask",
    "FWriteCLIParser",
    "TextIOTask",
    "TextIOCLIParser",
    "BinaryIOTask",
    "BinaryIOCLIParser",
    "QsortTask",
    "QsortCLIParser",
    "BsearchTask",
    "BsearchCLIParser",
    "ExitTask",
    "ExitCLIParser",
    "ExitFailureTask",
    "ExitFailureCLIParser",
    "AtexitTask",
    "AtexitCLIParser",
    "GetEnvTask",
    "GetEnvCLIParser",
    "SystemTask",
    "SystemCLIParser",

    "ErrnoTask",
    "ErrnoCLIParser",
    "PerrorTask",
    "PerrorCLIParser",
    "ErrorHandlingTask",
    "ErrorHandlingCLIParser",

    "NestedStructTask",
    "NestedStructCLIParser",
    "FlexibleArrayTask",
    "FlexibleArrayCLIParser",
    "BitFieldsTask",
    "BitFieldsCLIParser",
]