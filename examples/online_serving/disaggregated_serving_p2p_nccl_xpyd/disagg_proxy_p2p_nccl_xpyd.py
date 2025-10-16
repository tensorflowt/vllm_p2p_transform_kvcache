# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket
import threading
import time
import uuid
from typing import Any

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request

"""
工作流程
这些变量共同实现了分布式服务的动态发现和负载均衡机制：

节点通过ZMQ注册到对应的实例字典

条件变量确保线程安全的并发访问

计数器实现简单的轮询负载均衡

心跳机制自动清理失效节点

这些设计使得代理服务器能够动态管理后端服务节点，实现高可用性和负载均衡。
"""

count = 0 # 全局计数器，用于实现轮询负载均衡

# 表信息
# k:节点的HTTP地址（如 "192.168.1.100:8000"）
# v:元组 (ZMQ地址, 时间戳)
prefill_instances: dict[str, Any] = {}  # 预填充节点注册表 
decode_instances: dict[str, Any] = {}   # 解码节点注册表

prefill_cv = threading.Condition() # 预填充实例的线程同步条件变量
decode_cv = threading.Condition()  # 解码实例的线程同步条件变量

# 节点注册后5秒内需要重新发送心跳，否则会被清理
DEFAULT_PING_SECONDS = 5 # 心跳超时时间（秒）


def _remove_oldest_instances(instances: dict[str, Any]) -> None:
    """
    func:这是一个心跳超时清理函数，用于维护节点健康状态
    """
    oldest_key = next(iter(instances), None)
    while oldest_key is not None:
        value = instances[oldest_key]
        if value[1] > time.time():
            break
        print(f"🔴Remove [HTTP:{oldest_key}, ZMQ:{value[0]}, stamp:{value[1]}]")
        instances.pop(oldest_key, None)
        oldest_key = next(iter(instances), None)


def _listen_for_register(poller, router_socket):
    """
    func:持续监听并管理两类实例的生命周期
    """
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_address, message = router_socket.recv_multipart()
            # data: {"type": "P", "http_address": "ip:port",
            #        "zmq_address": "ip:port"}
            data = msgpack.loads(message)
            if data["type"] == "P":
                global prefill_instances
                global prefill_cv
                with prefill_cv:
                    node = prefill_instances.get(data["http_address"], None)
                    prefill_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(prefill_instances)

            elif data["type"] == "D":
                global decode_instances
                global decode_cv
                with decode_cv:
                    node = decode_instances.get(data["http_address"], None)
                    decode_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(decode_instances)
            else:
                print(
                    "Unexpected, Received message from %s, data: %s",
                    remote_address,
                    data,
                )
                return

            if node is None:
                print(f"🔵Add [HTTP:{data['http_address']}, ZMQ:{data['zmq_address']}]")


def start_service_discovery(hostname, port):
    """
    func: 初始化ZMQ通信框架并启动监听线程(服务发现机制的启动入口)。
    """
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")

    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    _listener_thread = threading.Thread(
        target=_listen_for_register, args=[poller, router_socket], daemon=True
    )
    _listener_thread.start()
    return _listener_thread

# 设置HTTP客户端超时配置（总超时时间为6小时）
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)


def random_uuid() -> str:
    """
    func: 生成简化版UUID,为每个请求创建唯一ID
    """
    return str(uuid.uuid4().hex)


async def forward_request(url, data, request_id):
    """
    func: 代理转发请求并支持流式响应,将请求转发到指定URL(预填充或解码节点)
    """
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                if True:
                    async for chunk_bytes in response.content.iter_chunked(1024):
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    """
    func: 异步HTTP请求处理器,支持两个OpenAI兼容的API端点,实现预填充和解码阶段的分离执行。
    """
    try:
        original_request_data = await request.get_json() # 获取原始请求
        prefill_request = original_request_data.copy()
        prefill_request["max_tokens"] = 1 # 限制只执行预填充
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1

        global count
        global prefill_instances
        global prefill_cv
        with prefill_cv:
            prefill_list = list(prefill_instances.items())
            prefill_addr, prefill_zmq_addr = prefill_list[count % len(prefill_list)] # 轮询选择算法
            prefill_zmq_addr = prefill_zmq_addr[0]

        global decode_instances
        global decode_cv
        with decode_cv:
            decode_list = list(decode_instances.items())
            decode_addr, decode_zmq_addr = decode_list[count % len(decode_list)]
            decode_zmq_addr = decode_zmq_addr[0]

        print(
            f"handle_request count: {count}, [HTTP:{prefill_addr}, "
            f"ZMQ:{prefill_zmq_addr}] 👉 [HTTP:{decode_addr}, "
            f"ZMQ:{decode_zmq_addr}]"
        )
        count += 1

        # 生成请求id
        request_id = (
            f"___prefill_addr_{prefill_zmq_addr}___decode_addr_"
            f"{decode_zmq_addr}_{random_uuid()}"
        )

        # 预填充阶段执行
        async for _ in forward_request(
            f"http://{prefill_addr}{request.path}", prefill_request, request_id
        ):
            continue

        # 解码阶段处理
        generator = forward_request(
            f"http://{decode_addr}{request.path}", original_request_data, request_id
        )
        response = await make_response(generator)
        response.timeout = None

        return response

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


if __name__ == "__main__":
    t = start_service_discovery("0.0.0.0", 30001)
    app.run(host="0.0.0.0", port=10001)
    t.join()