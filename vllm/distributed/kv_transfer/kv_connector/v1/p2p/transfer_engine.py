"""
created by yansiyu01@baidu.com at 2025/07/31
"""
import sys
import logging
import cuda_p2p_transfer

logger = logging.getLogger(__name__)


class P2PTransferEngine:
    """
    P2P传输引擎封装
    """
    def __init__(self, physical_gpu_id: int):
        """
        初始化P2P传输引擎
        """
        self.physical_gpu_id = physical_gpu_id
        try:
            self.p2p_engine = cuda_p2p_transfer.CudaP2PTransfer(physical_gpu_id)
            logger.info("P2P engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize P2P engine: {e}")
            raise RuntimeError(f"P2P engine initialization failed: {e}")
    
    def register_buffer(self, ptr):
        """
        注册缓冲区
        Args:
            ptr: 缓冲区指针
        Returns:
            缓冲区句柄 (bytes)
        """
        try:
            handle = self.p2p_engine.register_buffer(ptr)
            return handle
        except Exception as e:
            logger.error(f"Failed to register buffer at ptr={ptr}: {e}")
            raise RuntimeError(f"Buffer registration failed: {e}")

    def transfer(
        self,
        src_ptr: int,
        src_dev: int,
        dst_handle: bytes,
        dst_dev: int,
        dst_offset: int,
        length: int
    ):
        """
        执行P2P传输
        Args:
            src_ptr: 源指针
            src_dev: 源设备ID
            dst_handle: 目标句柄
            dst_dev: 目标设备ID
            dst_offset: 目标偏移量
            length: 传输长度
        Returns:
            0表示成功，非0表示错误
        """
        try:
            result = self.p2p_engine.transfer(
                src_ptr, 
                src_dev,
                dst_handle,
                dst_dev,
                dst_offset,
                length
            )
            
            if result != 0:
                logger.error(f"P2P transfer failed with code {result}")
            else:
                logger.debug("P2P transfer completed successfully")
            
            return result
        except Exception as e:
            logger.exception(f"Transfer failed: {e}")
            return 1
        
    def register_d_handle(self, dst_handle: bytes) -> int:
        """
        注册目标句柄到引擎
        Args:
            dst_handle: 目标句柄
        Returns:
            0表示成功，非0表示错误
        """
        try:
            result = self.p2p_engine.register_d_handle(dst_handle)
            if result != 0:
                logger.error(f"Failed to register destination handle {dst_handle.hex()} with code {result}")
            return result
        except Exception as e:
            logger.exception(f"Destination handle registration failed: {e}")
            return 1