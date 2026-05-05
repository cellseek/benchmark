from .datasets import (
    CellposeAdapter,
    BBBC038Adapter,
    CTCCellTrackingAdapter,
)
from .models.cellseek_adapter import CellSeekAdapter
from .models.cellpose_adapter import CellposeAdapter as CellposeModelAdapter
from .models.sam3_adapter import SAM3Adapter
from .models.microsam_adapter import MicroSAMAdapter
from .models.trackastra_adapter import TrackastraAdapter
from .models.ultrack_adapter import UltrackAdapter
from .models.celltractr_adapter import CelltractrAdapter


DATASET_REGISTRY = {
    "cellpose": CellposeAdapter,
    "bbbc038": BBBC038Adapter,
    "ctc_cell_tracking": CTCCellTrackingAdapter,
}


MODEL_REGISTRY = {
    "cellseek": CellSeekAdapter,
    "cellpose": CellposeModelAdapter,
    "sam3": SAM3Adapter,
    "microsam": MicroSAMAdapter,
    "trackastra": TrackastraAdapter,
    "ultrack": UltrackAdapter,
    "celltractr": CelltractrAdapter,
}
