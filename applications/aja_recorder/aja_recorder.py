# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from argparse import ArgumentParser

from holoscan.core import Application
from holoscan.operators import (
    AJASourceOp,
    FormatConverterOp,
    HolovizOp,
    VideoStreamRecorderOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    UnboundedAllocator,
)


class AJARecorder(Application):
    def __init__(self):
        """Initialize the endoscopy tool tracking application

        Parameters
        ----------
        record_type : {None, "input", "visualizer"}, optional
            Set to "input" if you want to record the input video stream, or
            "visualizer" if you want to record the visualizer output.
        source : {"replayer", "aja", "deltacast", "yuan"}
            When set to "replayer" (the default), pre-recorded sample video data is
            used as the application input. Otherwise, the video stream from an AJA or Yuan
            capture card is used.
        """
        super().__init__()

        # set name
        self.name = "AJA Recorder App"

        # Optional parameters affecting the graph created by compose.
        self.record_type = "input"
        if record_type is not None:
            if record_type not in ("input", "visualizer"):
                raise ValueError("record_type must be either ('input' or 'visualizer')")
        self.source = "aja"

    def compose(self):
        rdma = False

        if self.source.lower() == "aja":
            aja_kwargs = self.kwargs("aja")
            source = AJASourceOp(self, name="aja", **aja_kwargs)

            # 4 bytes/channel, 4 channels
            width = aja_kwargs["width"]
            height = aja_kwargs["height"]
            rdma = aja_kwargs["rdma"]
            source_block_size = width * height * 4 * 4
            source_num_blocks = 3 if rdma else 4
        else:
            raise ValueError("source must be 'aja'")

        source_pool_kwargs = dict(
            storage_type=MemoryStorageType.DEVICE,
            block_size=source_block_size,
            num_blocks=source_num_blocks,
        )
        assert self.record_type == "input"

        recorder_format_converter = FormatConverterOp(
            self,
            name="recorder_format_converter",
            pool=BlockMemoryPool(self, name="pool", **source_pool_kwargs),
            **self.kwargs("recorder_format_converter"),
        )
        recorder = VideoStreamRecorderOp(name="recorder", fragment=self, **self.kwargs("recorder"))

        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        visualizer_allocator = None

        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=width,
            height=height,
            enable_render_buffer_input=False,
            enable_render_buffer_output=False,
            allocator=visualizer_allocator,
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("holoviz"),
        )

        # Flow definition
        self.add_flow(
            source,
            visualizer,
            {("video_buffer_output", "receivers")},
        )
        self.add_flow(
            source,
            recorder_format_converter,
            {("video_buffer_output", "source_video")},
        )
        self.add_flow(recorder_format_converter, recorder)


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="AJA Recorder application.")
    parser.add_argument(
        "-r",
        "--record_type",
        choices=["none", "input", "visualizer"],
        default="none",
        help="The video stream to record (default: %(default)s).",
    )
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer", "aja", "yuan"],
        default="replayer",
        help=(
            "If 'replayer', replay a prerecorded video. Otherwise use a "
            "capture card as the source (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )
    args = parser.parse_args()
    record_type = args.record_type
    if record_type == "none":
        record_type = None

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "aja_recorder.yaml")
    else:
        config_file = args.config

    app = AJARecorder()
    app.config(config_file)
    app.run()
