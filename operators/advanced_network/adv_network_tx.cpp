/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "adv_network_tx.h"
#include "adv_network_dpdk_mgr.h"
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <memory>

namespace holoscan::ops {

struct AdvNetworkOpTx::AdvNetworkOpTxImpl {
  DpdkMgr *dpdk_mgr;
  struct rte_ring *tx_ring;
  struct rte_mempool *tx_meta_pool;
  AdvNetConfigYaml cfg;
};

struct UDPPkt {
  struct rte_ether_hdr eth;
  struct rte_ipv4_hdr ip;
  struct rte_udp_hdr udp;
  uint8_t payload[];
} __attribute__((packed));

void AdvNetworkOpTx::setup(OperatorSpec& spec)  {
  spec.input<AdvNetBurstParams *>("burst_in");

  spec.param(
      cfg_,
      "cfg",
      "Configuration",
      "Configuration for the advanced network operator",
      AdvNetConfigYaml());
}

void AdvNetworkOpTx::initialize() {
  HOLOSCAN_LOG_INFO("AdvNetworkOpTx::initialize()");
  register_converter<holoscan::ops::AdvNetConfigYaml>();

  holoscan::Operator::initialize();
  Init();
}

int AdvNetworkOpTx::Init() {
  impl = new AdvNetworkOpTxImpl();
  impl->cfg = cfg_.get();
  impl->dpdk_mgr = &dpdk_mgr;
  impl->tx_ring = nullptr;
  impl->dpdk_mgr->SetConfigAndInitialize(impl->cfg);

  // Set up all LUTs for speed
  for (auto &tx : cfg_.get().tx_) {
    auto port_opt = adv_net_get_port_from_ifname(tx.if_name_);
    if (!port_opt) {
      HOLOSCAN_LOG_CRITICAL("Failed to get port ID from interface {}", tx.if_name_);
      return -1;
    }
  }

  return 0;
}

void AdvNetworkOpTx::compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
      [[maybe_unused]] ExecutionContext&) {
  int n;

  if (unlikely(impl->tx_ring == nullptr)) {
    impl->tx_ring = rte_ring_lookup("TX_RING");
  }

  if (unlikely(impl->tx_meta_pool == nullptr)) {
    impl->tx_meta_pool = rte_mempool_lookup("TX_META_POOL");
  }

  auto burst = op_input.receive<AdvNetBurstParams *>("burst_in").value();

  AdvNetBurstParams *d_params;
  if (rte_mempool_get(impl->tx_meta_pool, reinterpret_cast<void**>(&d_params)) != 0) {
    HOLOSCAN_LOG_CRITICAL("Failed to get TX meta descriptor");
    return;
  }

  rte_memcpy(static_cast<void*>(d_params), burst, sizeof(*burst));
  if (rte_ring_enqueue(impl->tx_ring, reinterpret_cast<void *>(d_params)) != 0) {
    adv_net_free_tx_burst(burst);
    rte_mempool_put(impl->tx_meta_pool, d_params);
    HOLOSCAN_LOG_CRITICAL("Failed to enqueue TX work");
    return;
  }

  delete burst;
}
};  // namespace holoscan::ops