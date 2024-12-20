# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
# from .anymal_c.anymal import Anymal
# from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
# from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
# from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
# from .cassie.cassie import Cassie
# from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
# from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
# from .go_w.go_w import GO_W
# from .go_w.go_w_config import go_w_cfg, go_w_cfgppo
# from .a1w.a1w import A1W
# from .a1w.a1w_config import A1wFlatCfg, A1wFlatCfgPPO
from .go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO
# from .go1_fw.go1_fw import Go1Fw
# from .go1_fw.go1_fw_config import Go1FwFlatCfg, Go1FwFlatCfgPPO

from .go1_aw.go1_aw import Go1Aw
from .go1_aw.go1_aw_config import Go1AwFlatCfg, Go1AwFlatCfgPPO

from .go1_fw_clock.go1_fw import Go1FwClock
from .go1_fw_clock.go1_fw_config import Go1FwFlatClockCfg, Go1FwFlatClockCfgPPO

from .go1_legged.go1 import Go1_Flat
from .go1_legged.go1_config import Go1FlatCfg, Go1FlatCfgPPO
from .go1_legged.go1_id import Go1FlatID

# from .go1_fw_id.go1_id import Go1FwID
# from .go1_fw_id.go1_id_config import Go1FwFlatIDCfg, Go1FwFlatIDCfgPPO


from .go1_fw_tilt.go1_fw_tilt import Go1FwTilt
from .go1_fw_tilt.go1_fw_tilt_config import Go1FwFlatTiltCfg, Go1FwFlatTiltCfgPPO

from .go1_fw_terrain.go1_fw import Go1FwTerrain
from .go1_fw_terrain.go1_fw_config import Go1FwTerrainCfg, Go1FwTerrainCfgPPO

from .go1_fw_test.go1_fw import Go1FwTest
from .go1_fw_test.go1_fw_config import Go1FwTestCfg, Go1FwTestCfgPPO

# from .go1_corl.go1_corl import Go1CoRL
# from .go1_corl.go1_corl_config import Go1CoRLCfg, Go1CoRLCfgPPO

import os

from legged_gym.utils.task_registry import task_registry

# task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
# task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
# task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
# task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
# task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
# task_registry.register( "go_w", GO_W, go_w_cfg(), go_w_cfgppo())
# task_registry.register( "a1_w", A1W, A1wFlatCfg(), A1wFlatCfgPPO())
task_registry.register( "go1", LeggedRobot, Go1RoughCfg(), Go1RoughCfgPPO() )
# task_registry.register( "go1_fw", Go1Fw, Go1FwFlatCfg(), Go1FwFlatCfgPPO() )
task_registry.register( "go1_aw", Go1Aw, Go1AwFlatCfg(), Go1AwFlatCfgPPO() )
task_registry.register( "go1_fw_clock", Go1FwClock, Go1FwFlatClockCfg(), Go1FwFlatClockCfgPPO())
task_registry.register( "go1_legged", Go1_Flat, Go1FlatCfg(), Go1FlatCfgPPO())
task_registry.register( "go1_legged_id", Go1FlatID, Go1FlatCfg(), Go1FlatCfgPPO())

# task_registry.register( "go1_id", Go1FwID, Go1FwFlatIDCfg(), Go1FwFlatIDCfgPPO())
task_registry.register( "go1_fw_tilt", Go1FwTilt, Go1FwFlatTiltCfg(), Go1FwFlatTiltCfgPPO())
task_registry.register( "go1_fw_terrain", Go1FwTerrain, Go1FwTerrainCfg(), Go1FwTerrainCfgPPO())

task_registry.register( "go1_fw_test", Go1FwTest, Go1FwTestCfg(), Go1FwTestCfgPPO())

# task_registry.register( "go1_corl", Go1CoRL, Go1CoRLCfg(), Go1CoRLCfgPPO())