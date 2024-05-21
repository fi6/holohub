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
    V4L2VideoCaptureOp,
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


class VideoRecorder(Application):
    def __init__(self, source="v4l2"):
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
        self.name = "Video Recorder App"

        # Optional parameters affecting the graph created by compose.
        self.record_type = "input"
        if record_type is not None:
            if record_type not in ("input", "visualizer"):
                raise ValueError("record_type must be either ('input' or 'visualizer')")
        self.source = source
        self.video_dir = self.kwargs("recorder").get("directory", "../data/video_recorder")
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

    def compose(self):
        unbounded_pool = UnboundedAllocator(self, name="pool")

        if self.source.lower() == "aja":
            aja_kwargs = self.kwargs("aja")
            source = AJASourceOp(self, name="aja", **aja_kwargs)

            # 4 bytes/channel, 4 channels
            width = aja_kwargs["width"]
            height = aja_kwargs["height"]
            rdma = aja_kwargs["rdma"]
            source_block_size = width * height * 4 * 4
            source_num_blocks = 3 if rdma else 4
            output_label = "video_buffer_output"
        elif self.source.lower() == "v4l2":
            v4l2_kwargs = self.kwargs("v4l2")
            width = v4l2_kwargs.get("width", 1920)
            height = v4l2_kwargs.get("height", 1080)
            source = V4L2VideoCaptureOp(self, name="v4l2", allocator=unbounded_pool, **v4l2_kwargs)
            output_label = "signal"
        elif self.source.lower() == "replayer":
            width = 1920
            height = 1080
            source = VideoStreamReplayerOp(
                self,
                name="replayer",
                directory=self.video_dir,
                **self.kwargs("replayer"),
            )
            output_label = "output"
        else:
            raise ValueError("source must be 'aja' or 'v4l2'")

        # source_pool_kwargs = dict(
        #     storage_type=MemoryStorageType.DEVICE,
        #     block_size=source_block_size,
        #     num_blocks=source_num_blocks,
        # )

        # recorder_format_converter = FormatConverterOp(
        #     self,
        #     name="recorder_format_converter",
        #     pool=BlockMemoryPool(self, name="pool", **source_pool_kwargs),
        #     **self.kwargs("recorder_format_converter"),
        # )

        recorder_format_converter = FormatConverterOp(
            self,
            name="recorder_format_converter",
            pool=unbounded_pool,
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
            {(output_label, "receivers")},
        )
        self.add_flow(
            source,
            recorder_format_converter,
            {(output_label, "source_video")},
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
        choices=["v4l2", "aja", "replayer"],
        default="v4l2",
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
        config_file = os.path.join(os.path.dirname(__file__), "video_recorder.yaml")
    else:
        config_file = args.config

    app = VideoRecorder(source=args.source)
    app.config(config_file)
    app.run()
