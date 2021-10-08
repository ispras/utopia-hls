/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file device_api.cc
 * \brief TVM device API for VTA
 */

#include <device_api.h>
#include <runtime.h>

namespace tvm {
namespace runtime {

class VTADeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(DLDevice dev) final {}

  void* AllocDataSpace(DLDevice dev, size_t size, size_t alignment, DLDataType type_hint) final {
    return VTABufferAlloc(size);
  }

  void FreeDataSpace(DLDevice dev, void* ptr) final { VTABufferFree(ptr); }

  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      DLDevice dev_from, DLDevice dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    int kind_mask = 0;
    if (dev_from.device_type != kDLCPU) {
      kind_mask |= 2;
    }
    if (dev_to.device_type != kDLCPU) {
      kind_mask |= 1;
    }
    VTABufferCopy(from, from_offset, to, to_offset, size, kind_mask);
  }

  void StreamSync(DLDevice dev, TVMStreamHandle stream) final {}

  void* AllocWorkspace(DLDevice dev, size_t size, DLDataType type_hint) final;

  void FreeWorkspace(DLDevice dev, void* data) final;

  static VTADeviceAPI* Global() {
    static VTADeviceAPI* inst = new VTADeviceAPI();
    return inst;
  }
};

}  // namespace runtime
}  // namespace tvm
