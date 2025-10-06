# models package
# ensure side-effect imports register models
from . import template_matching  # noqa: F401
# CSRT 已停用，如需重新啟用請取消下行註解
# from . import csrt  # noqa: F401
from . import optical_flow_lk  # noqa: F401
from . import faster_rcnn  # noqa: F401
from . import yolov11  # noqa: F401
from . import fast_speckle  # noqa: F401
from . import ocsort  # noqa: F401
from . import strongsort  # noqa: F401
