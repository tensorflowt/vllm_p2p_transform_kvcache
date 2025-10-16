import time  
import threading  
from dataclasses import dataclass  
from typing import TYPE_CHECKING, Any, Optional  
  
import msgpack  
import torch  
import zmq  
import regex as re  
  
from vllm.config import VllmConfig  
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)  
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.transfer_engine import (  
    P2PTransferEngine)  
from vllm.distributed.parallel_state import get_world_group  
from vllm.logger import init_logger  
from vllm.utils import get_ip  
from vllm.v1.attention.backends.mla.common import MLACommonMetadata  
  
if TYPE_CHECKING:  
    from vllm.attention.backends.abstract import AttentionMetadata  
    from vllm.forward_context import ForwardContext  
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks  
    from vllm.v1.request import Request  
    from vllm.v1.core.sched.output import SchedulerOutput  
  
logger = init_logger(__name__)  
  
  
@dataclass  
class CustomP2pMetadata(KVConnectorMetadata):  
    """元数据结构,包含请求信息和句柄"""  
    requests: list[dict[str, Any]]  
  
    def __init__(self):  
        self.requests = []  
  
  
class CustomP2pConnector(KVConnectorBase_V1):  
    """基于自定义P2P传输引擎和ZMQ的KV连接器"""  
  
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):  
        super().__init__(vllm_config=vllm_config, role=role)  
  
        self.config = vllm_config.kv_transfer_config  
        self.is_producer = self.config.is_kv_producer  
        self._block_size = vllm_config.cache_config.block_size  
  
        self._rank = get_world_group().rank \
            if role == KVConnectorRole.WORKER else 0  
        self._local_rank = get_world_group().local_rank \
            if role == KVConnectorRole.WORKER else 0  
  
        # 初始化P2P引擎 (仅在Worker端)  
        self.p2p_engine = P2PTransferEngine(  
            physical_gpu_id=self._local_rank  
        ) if role == KVConnectorRole.WORKER else None  
  
        # 初始化remote_handles和socks  
        self.remote_handles: dict[str, bytes] = {}  
        self.socks: dict[str, Any] = {}  
          
        # Scheduler端:记录需要加载的请求  
        self._requests_need_load: dict[str, tuple["Request", "KVCacheBlocks"]] = {}  
  
        # ZMQ通信设置 (仅在Worker端)  
        if role == KVConnectorRole.WORKER:  
            logger.info("Role is WORKER, setting up ZMQ") 
            self._setup_zmq()  
        else:
            logger.info("Role is %s, skipping ZMQ setup", role)
  
    def _setup_zmq(self):  
        """设置ZMQ服务器用于句柄交换"""  
        try:
            hostname = get_ip()  
            port = int(self.config.kv_port) + self._rank  
            logger.info("Setting up ZMQ on %s:%d", hostname, port) 
    
            self.zmq_address = f"{hostname}:{port}"  
            self.http_address = f"{hostname}:{self.config.kv_connector_extra_config['http_port']}"  
    
            # 获取proxy地址  
            proxy_ip = self.config.get_from_extra_config("proxy_ip", "")  
            proxy_port = self.config.get_from_extra_config("proxy_port", "")  
            if proxy_ip == "" or proxy_port == "":  
                self.proxy_address = ""  
            else:  
                self.proxy_address = proxy_ip + ":" + proxy_port  
    
            # 创建ZMQ context和ROUTER socket  
            self.context = zmq.Context()  
            self.router_socket = self.context.socket(zmq.ROUTER)  
            self.router_socket.bind(f"tcp://0.0.0.0:{port}")  
    
            self.poller = zmq.Poller()  
            self.poller.register(self.router_socket, zmq.POLLIN)  
    
            # 启动监听线程  
            self._listener_thread = threading.Thread(  
                target=self._listen_for_requests, daemon=True)  
            self._listener_thread.start()  
    
            # 启动心跳线程(仅rank0发送)  
            self._ping_thread = None  
            if self._rank == 0 and self.proxy_address != "":  
                self._ping_thread = threading.Thread(target=self._ping, daemon=True)  
                self._ping_thread.start()  
    
            logger.info(  
                "CustomP2pConnector initialized, rank:%d, local_rank:%d, "  
                "zmq_address:%s, proxy_address:%s",  
                self._rank, self._local_rank, self.zmq_address, self.proxy_address)  
        except Exception as e:  
            logger.error("Failed to setup ZMQ: %s", str(e), exc_info=True)  
            raise

  
    def _listen_for_requests(self):    
        """监听 ZMQ 请求,处理句柄交换"""    
        logger.info("start listen for requests!")  
        while True:    
            try:    
                socks = dict(self.poller.poll())  
                
                # 添加调试日志  
                logger.info("Poll returned %d sockets", len(socks))  
                
                if self.router_socket not in socks:    
                    continue  
                
                logger.info("Received message on router_socket")  
    
                # ROUTER socket 接收多帧消息: [identity, empty_delimiter, message]    
                *identity_frames, message = self.router_socket.recv_multipart()    
                data = msgpack.loads(message)    
                logger.info("Received data: %s", data)  
    
                if data["cmd"] == "REGISTER_HANDLE":    
                    # 接收远程句柄注册请求    
                    handle_id = data["handle_id"]    
                    handle_bytes = bytes(data["handle"])    
                    self.remote_handles[handle_id] = handle_bytes    
    
                    # 注册到 P2P 引擎    
                    if self.p2p_engine:    
                        logger.info("Registering remote handle: %s", handle_id)  
                        result = self.p2p_engine.register_d_handle(handle_bytes)    
                        # 回复确认    
                        self.router_socket.send_multipart(    
                            identity_frames + [msgpack.dumps({"ret": result})])    
                    else:  
                        logger.warning("p2p_engine is None, cannot register handle")  
    
                    logger.info("Registered remote handle: %s (total: %d)",     
                            handle_id, len(self.remote_handles))    
                    logger.debug("Current remote_handles: %s", list(self.remote_handles.keys()))  
                else:  
                    logger.warning("Unknown command: %s", data.get("cmd"))  
                    
            except Exception as e:    
                logger.error("Error in ZMQ listener: %s", str(e), exc_info=True) 
  
    def _ping(self):  
        """定期向 Proxy 发送心跳包进行服务注册"""  
        sock = self.context.socket(zmq.DEALER)  
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)  
        logger.debug("ping start, zmq_address:%s", self.zmq_address)  
        sock.connect(f"tcp://{self.proxy_address}")  
  
        data = {  
            "type": "P" if self.config.is_kv_producer else "D",  
            "http_address": self.http_address,  
            "zmq_address": self.zmq_address  
        }  
  
        # 序列化一次,避免重复序列化  
        serialized_data = msgpack.dumps(data)  
  
        while True:  
            try:  
                sock.send(serialized_data)  
                logger.debug("Sent ping to proxy: %s", self.proxy_address)  
            except Exception as e:  
                logger.warning("Failed to send ping: %s", str(e))  
            time.sleep(3)  
  
    def _create_connection(self, remote_address: str) -> Any:  
        """创建到远程地址的 ZMQ 连接"""  
        if remote_address not in self.socks:  
            sock = self.context.socket(zmq.DEALER)  
            sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)  
            sock.connect(f"tcp://{remote_address}")  
            self.socks[remote_address] = sock  
            logger.info("ZMQ connection to %s established", remote_address)  
        return self.socks[remote_address]  
  
    def _register_local_buffer(self, buffer_ptr: int) -> bytes:  
        """注册本地buffer并获取句柄"""  
        if self.p2p_engine is None:  
            return b''  
        return self.p2p_engine.register_buffer(buffer_ptr)  
  
    def _exchange_handle(self, remote_address: str, local_handle: bytes,  
                        handle_id: str) -> None:  
        """与远程节点交换句柄"""  
        logger.info("Attempting to connect to Producer at %s", remote_address)  
        sock = self._create_connection(remote_address)  
  
        # 发送本地句柄到远程  
        data = {  
            "cmd": "REGISTER_HANDLE",  
            "handle_id": handle_id,  
            "handle": list(local_handle)  
        }  
        logger.info("Sending handle %s to %s", handle_id, remote_address)  
        sock.send(msgpack.dumps(data))  
  
        # 等待确认  
        try:  
            response = sock.recv(flags=zmq.NOBLOCK)  
            result = msgpack.loads(response)  
  
            if result.get("ret", 0) != 0:  
                logger.error("Failed to register handle at remote: %s", remote_address)  
            else:  
                logger.info("Successfully registered handle %s at %s", handle_id, remote_address)  
        except zmq.Again:  
            # 非阻塞接收,如果没有响应就继续  
            pass  
  
    def _extract_kv(self, layer: torch.Tensor, block_ids: torch.Tensor,  
                   attn_metadata: "AttentionMetadata") -> Optional[torch.Tensor]:  
        """从层中提取 KV cache"""  
        if isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2:  
            return layer[block_ids, ...]  
        elif layer.shape[0] == 2:  
            return layer[:, block_ids, ...]  
        return None  
  
    def parse_request_id(self, request_id: str, is_producer: bool) -> tuple[str, int]:  
        """解析 request_id 获取 IP 和端口"""  
        # 格式: cmpl-___prefill_addr_IP:PORT___decode_addr_IP:PORT_UUID-N  
        if is_producer:  
            match = re.search(r'decode_addr_([\d\.]+):(\d+)', request_id)  
        else:  
            match = re.search(r'prefill_addr_([\d\.]+):(\d+)', request_id)  

        if match:  
            ip = match.group(1)  
            port = int(match.group(2))  
            logger.info("Parsed request_id: %s -> %s:%d", request_id, ip, port)  
            return ip, port  
        raise ValueError(f"Invalid request_id format: {request_id}")  
  
    # ==============================  
    # Worker-side methods  
    # ==============================  
  
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:  
        """Worker端:开始加载 KV cache,执行buffer注册和句柄交换"""  
        
        
        if self.is_producer or self.p2p_engine is None:  
            return  
    
        metadata = self._get_connector_metadata()  
        logger.info("[%s] start_load_kv called, requests: %s",   
            time.time(), [r['request_id'] for r in metadata.requests])  
        if not isinstance(metadata, CustomP2pMetadata):  
            return  
    
        # 记录已经注册过句柄的请求  
        if not hasattr(self, '_registered_handles'):  
            self._registered_handles = set()  
    
        for req_info in metadata.requests:  
            request_id = req_info['request_id']  
            
            # 跳过已经注册过的请求  
            if request_id in self._registered_handles:  
                continue  
            
            block_ids = req_info['block_ids']  
            if not block_ids:  
                logger.warning("No blocks for request %s", request_id)  
                continue  
    
            try:  
                # 遍历所有 attention layers 来获取 KV cache  
                for layer_name in forward_context.no_compile_layers:  
                    layer = forward_context.no_compile_layers[layer_name]  
                    kv_cache_attr = getattr(layer, 'kv_cache', None)  
                    if kv_cache_attr is None:  
                        continue  
    
                    kv_cache_layer = kv_cache_attr[forward_context.virtual_engine]  
                    buffer_ptr = kv_cache_layer.data_ptr()  
                    
                    # 注册 buffer 获取句柄  
                    local_handle = self._register_local_buffer(buffer_ptr)  
                    
                    if not local_handle:  
                        logger.error("Failed to register local buffer for request %s", request_id)  
                        break  
    
                    # 解析 Producer 地址  
                    ip, port = self.parse_request_id(request_id, is_producer=False)  
                    producer_address = f"{ip}:{port}"  
    
                    # 通过 ZMQ 发送句柄给 Producer  
                    handle_id = request_id  
                    self._exchange_handle(producer_address, local_handle, handle_id)  
    
                    logger.info("Consumer Worker: Registered and sent handle %s to producer %s",  
                            handle_id, producer_address)  
                    
                    # 标记为已注册  
                    self._registered_handles.add(request_id)  
                    break  
    
            except Exception as e:  
                logger.error("Failed to register buffer for request %s: %s",  
                            request_id, str(e)) 
  
    def wait_for_layer_load(self, layer_name: str) -> None:  
        """等待层加载完成"""  
        # 传输是同步的,无需等待  
        pass  
  
    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,  
                     attn_metadata: "AttentionMetadata", **kwargs: Any) -> None:  
        """Producer Worker端:保存 KV cache 层"""  
        logger.info("start save kv layer!")
        
        if not self.is_producer or self.p2p_engine is None:  
            return  
    
        metadata = self._get_connector_metadata()  
        if not isinstance(metadata, CustomP2pMetadata):  
            return  
    
        # 在循环外部的日志不应该引用 request_id  
        logger.debug("save_kv_layer called for layer %s with %d requests",   
                    layer_name, len(metadata.requests))  
    
        for req_info in metadata.requests:  
            request_id = req_info['request_id']  # 在这里定义  
            
            # 现在可以安全地使用 request_id  
            logger.debug("[%s] Processing request %s in save_kv_layer",   
                        time.time(), request_id)  
            
            block_ids = torch.tensor(req_info['block_ids'])  
            
            # 实时从 remote_handles 获取最新的句柄  
            handle_id = request_id  
            dst_handle = self.remote_handles.get(handle_id, b'')  
            
            # 如果句柄仍然为空,尝试等待一段时间  
            if not dst_handle:  
                logger.warning("No dst_handle for request %s, waiting...", request_id)  
                max_retries = 10  
                retry_interval = 0.1  
                
                for i in range(max_retries):  
                    time.sleep(retry_interval)  
                    dst_handle = self.remote_handles.get(handle_id, b'')  
                    if dst_handle:  
                        logger.info("Got dst_handle for request %s after %d retries",   
                                request_id, i + 1)  
                        break  
                
                if not dst_handle:  
                    logger.error("Failed to get dst_handle for request %s after retries",   
                            request_id)  
                    continue  
            
            dst_dev = req_info.get('dst_dev', 0)  
    
            # 提取 KV cache  
            kv_cache = self._extract_kv(kv_layer, block_ids, attn_metadata)  
            if kv_cache is None:  
                continue  
    
            # 执行 P2P 传输  
            try:  
                handle = self.p2p_engine.transfer(  
                    src_ptr=kv_cache.data_ptr(),  
                    src_dev=self._local_rank,  
                    dst_handle=dst_handle,  
                    dst_dev=dst_dev,  
                    dst_offset=0,  
                    length=kv_cache.numel() * kv_cache.element_size()  
                )  
    
                # 等待传输完成  
                while not handle.done:  
                    time.sleep(0.001)  
    
                logger.debug("Transfer completed for %s#%s", request_id, layer_name)  
            except Exception as e:  
                logger.error("Transfer failed for %s#%s: %s",  request_id, layer_name, str(e))
  
    def wait_for_save(self) -> None:  
        """等待保存完成"""  
        # 传输在 save_kv_layer 中已同步完成  
        pass  
  
    # ==============================  
    # Scheduler-side methods  
    # ==============================  
  
    def get_num_new_matched_tokens(  
        self,  
        request: "Request",  
        num_computed_tokens: int,  
    ) -> tuple[Optional[int], bool]:  
        """获取匹配的 token 数量"""  
        # 简单实现:不支持 KV cache 复用  
        if self.is_producer:  
            return 0, False  
  
        # Consumer端:从 producer 加载所有 prompt tokens  
        num_external_tokens = (len(request.prompt_token_ids) - 1 -  
                              num_computed_tokens)  
  
        if num_external_tokens < 0:  
            num_external_tokens = 0  
  
        return num_external_tokens, False  
  
    def update_state_after_alloc(self, request: "Request",  
                                 blocks: "KVCacheBlocks",  
                                 num_external_tokens: int):  
        """Scheduler端:记录需要加载的请求"""  
        if self.is_producer:  
            return  
          
        # 只在 Scheduler 端记录需要加载的请求  
        if num_external_tokens > 0:  
            self._requests_need_load[request.request_id] = (request, blocks)  
            logger.info("Scheduler: Recorded request %s for loading (%d external tokens)",  
                       request.request_id, num_external_tokens)  
  
    def build_connector_meta(  
        self, scheduler_output: "SchedulerOutput", **kwargs  
    ) -> KVConnectorMetadata:  
        """构建连接器元数据"""  
        metadata = CustomP2pMetadata()  
  
        for request in scheduler_output.scheduled_new_reqs:  
            request_id = request.req_id  
  
            # 解析远程地址  
            try:  
                ip, port = self.parse_request_id(request_id, self.is_producer)  
                remote_address = f"{ip}:{port}"  
            except ValueError:  
                logger.warning("Failed to parse request_id: %s", request_id)  
                continue  
  
            # Producer端: 尝试获取句柄,但不阻塞等待  
            dst_handle = b''  
            if self.is_producer:  
                handle_id = request_id  
                dst_handle = self.remote_handles.get(handle_id, b'')  
                if not dst_handle:  
                    logger.debug("No dst_handle yet for request %s, will retry in save_kv_layer",   
                               request_id)  
  
            metadata.requests.append({  
                'request_id': request_id,  
                'block_ids': request.block_ids[0] if request.block_ids else [],  
                'dst_handle': dst_handle,  
                'dst_dev': self._local_rank,  
                'remote_address': remote_address  
            })  
  
        return metadata  
  
    def request_finished(  
        self,  
        request: "Request",  
        block_ids: list[int],  
    ) -> tuple[bool, Optional[dict[str, Any]]]:  
        """请求完成时的处理"""  
        # 清理记录的请求  
        if request.request_id in self._requests_need_load:  
            del self._requests_need_load[request.request_id]  
          
        # 返回 False 表示可以立即释放 KV cache  
        return False, None