# Copyright (C) 2016-2018 Alibaba Group Holding Limited
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
import os
import xdl
import subprocess
from xdl.python import pybind
from xdl.python.io.data_io import DataIO
from xdl.python.io.data_sharding import DataSharding

class DataReader(DataIO):
    def __init__(self, ds_name, file_type=pybind.parsers.txt,
                 fs_type = None,
                 namenode="",
                 paths=None,
                 meta=None,
                 enable_state=True):
        self._ds_name = ds_name
        self._paths = list()
        self._meta = meta
        self._fs_type = fs_type
        self._namenode = namenode

        if paths is not None:
            assert isinstance(paths, list), "paths must be a list"

            for path in paths:
                fs_type, namenode, rpath = self._decode_path(path)

                if self._fs_type is not None and fs_type is not None:
                    assert fs_type == self._fs_type, "support only one filesystem %s"%self._fs_type
                else:
                    self._fs_type = fs_type

                if self._namenode != "" and namenode != "":
                    assert namenode == self._namenode, "support only one namenode %s"%self._namenode
                else:
                    self._namenode = namenode

                if rpath is not None:
                    self._paths.append(rpath)

        if self._fs_type is None:
            self._fs_type = pybind.fs.local

        super(DataReader, self).__init__(ds_name, file_type=file_type,
                                         fs_type=self._fs_type, namenode=self._namenode,
                                         enable_state=enable_state)

        # add path after failover
        self._sharding = DataSharding(self.fs())
        self._sharding.add_path(self._paths)

        paths = self._sharding.partition(
            rank=xdl.get_task_index(), size=xdl.get_task_num())
        print('data paths:', paths)
        self.add_path(paths)
        if self._meta is not None:
            self.set_meta(self._meta)
    def _exec_cmd(self, exec_cmd):
        p = subprocess.Popen(exec_cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
        return p
    def _decode_path(self, path):
        '''
        hdfs://namenode/path
        kafka://namenode
        '''
        namenode = ""
        fs_type = None
        fpath = None
        if path.startswith('hdfs://'):
            fs_type = pybind.fs.hdfs
            arr = path.split('/', 3)
            assert len(arr) == 4
            namenode = arr[2]
            fpath = '/'+arr[3]
            if arr[3].endswith(".gz"):
                currentDirectory = os.getcwd()
                hdp_cmd ="hadoop fs -get {} {}".format(path, currentDirectory)
                self._exec_cmd(hdp_cmd)
                dirs_splited = path.split('/')
                gz_filename = dirs_splited[len(dirs_splited)-1]

                gzip_cmd = "gunzip {}/{}".format(currentDirectory,gz_filename)
                self._exec_cmd(gzip_cmd)

                
                local_path = "{}/{}".format(currentDirectory,gz_filename.strip(".gz"))
                
                script_cmd = "python {}/main_newData.py {}".format(currentDirectory,local_path)
                self._exec_cmd(script_cmd)
                
                fpath =  local_path+".txt"
                fs_type = pybind.fs.local
                
        elif path.startswith('kafka://'):
            fs_type = pybind.fs.kafka
            arr = path.split('/', 2)
            assert len(arr) == 3
            namenode = arr[2]
        else:
            assert '://' not in path, "Unsupported path: %s" % path
            fpath = path

        return fs_type, namenode, fpath

