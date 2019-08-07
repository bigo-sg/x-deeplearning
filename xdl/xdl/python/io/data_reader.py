from datetime import datetime
import sys
import os
import xdl
import subprocess
from xdl.python import pybind
from xdl.python.io.data_io import DataIO
from xdl.python.io.data_sharding import DataSharding
import threading


threadLock = threading.Lock()
threads = []
class myThread (threading.Thread):
   def __init__(self, threadID, inputlist,  converter_file):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.input = inputlist
      self._converter_file = converter_file
   def run(self):
      print "Starting " + self.name+ " tid "+ str(self.threadID)
      for i in range(len(self.input[self.threadID])):
         print self.input[self.threadID][i]+"  "+str(self.threadID)
         file_parsing(self.input[self.threadID][i],self._converter_file,self.threadID)
     
def file_parsing(path,converter_file, tid):
     currentDirectory = os.getcwd()
     hdp_cmd ="hadoop fs -get {} {}".format(path, currentDirectory)
     dirs_splited = path.split('/')
     gz_filename = dirs_splited[len(dirs_splited)-1]
     rm_cmd = "rm -f {}/{}".format(currentDirectory,gz_filename)
     exec_cmd(rm_cmd)
     p=exec_cmd(hdp_cmd)
     if p==0:
        print "after hdp_cmd"+"  "+ str(tid)
     else:
        raise ValueError('hadoop download error')    
     rm_cmd = "rm -f {}/{}".format(currentDirectory,gz_filename.strip(".gz"))
     gzip_cmd = "gzip -d {}/{}".format(currentDirectory,gz_filename)
     exec_cmd(rm_cmd)
     exec_cmd(gzip_cmd)

     local_path = "{}/{}".format(currentDirectory,gz_filename.strip(".gz"))

     script_cmd = "python {}/{} {}".format(currentDirectory,converter_file,local_path)
     print script_cmd+" "+str(tid)
     p=exec_cmd(script_cmd)
     if p==0:
        print str(tid)+"finish task"
     else:
        raise ValueError('python parsing error')
def exec_cmd(exec_cmd1):
     p = subprocess.call(exec_cmd1,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
     return p




class DataReader(DataIO):
    def __init__(self, ds_name, file_type=pybind.parsers.txt,
                 fs_type = None,
                 namenode="",
                 paths=None,
                 meta=None,
                 enable_state=True,
                 converter_file="", converter_threads=10):
        self._ds_name = ds_name
        self._paths = list()
        self._meta = meta
        self._fs_type = fs_type
        self._namenode = namenode
        self._converter_file = converter_file
        self.gz_hdfs_dir=""
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

        shard_paths = self._sharding.partition(
            rank=xdl.get_task_index(), size=xdl.get_task_num())
        converted_path=[]
        thread_input= []
        
        if self.gz_hdfs_dir!="":
          curr_thread=0
          thread_input.append([])
        else:
          curr_thread=-1
        threads_list=[]
        print "start"+"  "+str(datetime.now())
       
        for raw_path in shard_paths:
          arr = raw_path.split('/')
          filename = arr[-1]
          
          if filename.startswith('gz_hdfs_'):
             gz_hdfs_addr = self.gz_hdfs_dir+(filename.strip('gz_hdfs_')).strip('.txt')+'.gz'
             if len(thread_input[curr_thread])< ((len(shard_paths)/converter_threads)):
               thread_input[curr_thread].append(gz_hdfs_addr)
               #inputlist.append(gz_hdfs_addr)               
             else:
               if curr_thread < ((len(shard_paths)%converter_threads)): 
                  thread_input[curr_thread].append(gz_hdfs_addr)
                  curr_thread = curr_thread+1
                  thread_input.append([])
               else:
                  curr_thread = curr_thread+1
                  thread_input.append([])
                  thread_input[curr_thread].append(gz_hdfs_addr)
        for i in range(len(thread_input)):
           print str(i)+" thread len "+str(len(thread_input[i]))
        if curr_thread>=0:
          #thread_input.append(inputlist)
          for i in range(curr_thread+1):
            threads_list.append(myThread(i, thread_input,self._converter_file))
                
            threads_list[i].start()
          for i in range(curr_thread+1):
            threads_list[i].join()

          
        print "end"+"  "+str(datetime.now())      
        print('data paths:', shard_paths)
        for i in shard_paths:
          #st = os.stat(i)
          if ((curr_thread>=0)):
            st = os.stat(i)
            if (st.st_size<=0):
              print i+ "  size error"
        self.add_path(shard_paths)
       
        if self._meta is not None:
            self.set_meta(self._meta)
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
                arr2 = path.split('/')
                if self.gz_hdfs_dir =="":
                   self.gz_hdfs_dir=path.strip(arr2[-1])
                if self.gz_hdfs_dir!= path.strip(arr2[-1]):
                  print "error file dir"
                  return
                currentDirectory = os.getcwd()
                dirs_splited = path.split('/')
                gz_filename = dirs_splited[len(dirs_splited)-1]
                local_path = "{}/gz_hdfs_{}".format(currentDirectory,gz_filename.strip(".gz"))
                xdl_file = open(local_path+".txt", "w")
                xdl_file.close()
                
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
