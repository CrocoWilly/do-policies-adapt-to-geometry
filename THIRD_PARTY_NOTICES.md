# Third-party notices

This repository extends and redistributes material from the projects below. Each
is covered by its own license, reproduced in full here. The licenses are
permissive and allow redistribution; all of them require that the copyright
notice be retained, which is the purpose of this file.

---

## NVIDIA Isaac Lab

**BSD-3-Clause** · https://github.com/isaac-sim/IsaacLab

> Copyright (c) 2022-2025, The Isaac Lab Project Developers
> (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
> All rights reserved.

The `source/` and `scripts/` trees mirror Isaac Lab's directory layout and are
designed to be overlaid onto a checkout. Files that extend or modify Isaac Lab
code retain its original copyright header. The license text is reproduced in
[LICENSE](LICENSE), which this project shares.

---

## Techman TM7s robot description

**BSD-3-Clause** · https://github.com/TechmanRobotInc/tmr_ros2 (`tm_description`)

> Copyright (c) 2019-2026, TECHMAN ROBOT INC.
> All rights reserved.

The TM7s arm links, joints and visual meshes inside
`usd_assets/tm7s_robotiq85_forIsaacLab_flatten.usd` derive from Techman's
published robot description. The URDF was merged with a gripper model, converted
to USD and re-tuned for Isaac Lab; the underlying geometry is Techman's.

```
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## Robotiq 2F-85 gripper description

**BSD-2-Clause** · https://github.com/ros-industrial/robotiq
obtained via https://github.com/KevinGalassi/Robotiq-2f-85

> Copyright (c) 2013, ROS-Industrial
> All rights reserved.

The Robotiq 2F-85 links, joints and visual meshes inside
`usd_assets/tm7s_robotiq85_forIsaacLab_flatten.usd` derive from this package.
The gripper was attached to the TM7s flange, its mimic and closure joints
declared for the merged kinematic tree, and its actuator gains re-tuned for
Isaac Lab; the underlying geometry is ROS-Industrial's and Robotiq's.

```
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## Franka Emika Panda

The insertion experiments use the Franka Panda articulation that ships with
Isaac Lab; no Franka assets are redistributed here beyond what Isaac Lab
provides.

## Materials

Peg and table materials reference MDL files from the NVIDIA Nucleus asset
library at runtime (`{NVIDIA_NUCLEUS_DIR}/Materials/...`). Those assets are not
redistributed here and remain subject to NVIDIA's terms.
