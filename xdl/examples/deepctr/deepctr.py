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

import tensorflow as tf
import xdl

reader = xdl.DataReader("r1", # name of reader
                        paths=["./data.txt"], # file paths
                        enable_state=False) # enable reader state

reader.epochs(1).threads(1).batch_size(10).label_count(1)
reader.feature(name='sparse0', type=xdl.features.sparse)\
    .feature(name='sparse1', type=xdl.features.sparse)\
    .feature(name='deep0', type=xdl.features.dense, nvec=256)
reader.startup()

def train():
    batch = reader.read()
    sess = xdl.TrainSession()
    emb1 = xdl.embedding('emb1', batch['sparse0'], xdl.TruncatedNormal(stddev=0.001), 8, 1024, vtype='hash')
    emb2 = xdl.embedding('emb2', batch['sparse1'], xdl.TruncatedNormal(stddev=0.001), 8, 1024, vtype='hash')
    hooks = []

    vars = ["emb1", "emb2"]
    mark_hook1 = xdl.GlobalStepMarkHook("emb1", batch["sparse0"].ids)
    mark_hook2 = xdl.GlobalStepMarkHook("emb2", batch["sparse1"].ids)
    hooks.append(mark_hook1)
    hooks.append(mark_hook2)
    if xdl.get_task_index() == 0:
        #filter_hook = xdl.GlobalStepAndL2FilterHook(vars, 30, 10, 0.00001, " i>global_step || d>#L2#_ ")
        #filter_hook = xdl.GlobalStepAndL2FilterHook(vars, 30, 10, 0.00001, "i>global_step && d>#L2#_")
        filter_hook = xdl.GlobalStepAndL2FilterHook(vars, 30, 10, 0.002, "i>global_step&&d>#L2#_")
        hooks.append(filter_hook)
    loss = model(batch['deep0'], [emb1, emb2], batch['label'])
    train_op = xdl.SGD(0.5).optimize()
    log_hook = xdl.LoggerHook(loss, "loss:{0}", 10)
    hooks.append(log_hook)
    sess = xdl.TrainSession(hooks=hooks)
    while not sess.should_stop():
        sess.run(train_op)

@xdl.tf_wrapper()
def model(deep, embeddings, labels):
    input = tf.concat([deep] + embeddings, 1)
    fc1 = tf.layers.dense(
        input, 128, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    fc2 = tf.layers.dense(
        fc1, 64, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    fc3 = tf.layers.dense(
        fc2, 32, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    y = tf.layers.dense(
        fc3, 1, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    loss = tf.losses.sigmoid_cross_entropy(labels, y)
    return loss

train()        

