# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .fed_avg_g import PYUFedAvgG
from .fed_avg_u import PYUFedAvgU
from .fed_avg_w import PYUFedAvgW
from .fed_gen import PYUFedGen
from .fed_prox import PYUFedProx
from .fed_scr import PYUFedSCR
from .fed_stc import PYUFedSTC
from .moon import PYUFedMOON
from .scaffold import PYUScaffold
from .orchestra import PYUOrchestraStrategy, PYUOrchestraSimpleStrategy

__all__ = [
    'PYUFedAvgW',
    'PYUFedAvgG',
    'PYUFedAvgU',
    'PYUFedProx',
    'PYUFedSCR',
    'PYUFedSTC',
    'PYUScaffold',
    'PYUFedGen',
    'PYUFedMOON',
    'PYUOrchestraStrategy',
    'PYUOrchestraSimpleStrategy',
]
