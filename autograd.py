"""
基础自动微分框架 (Basic Autograd Framework)
支持标量和向量的自动微分计算
"""

import numpy as np
from typing import Optional, List, Union, Callable


class Variable:
    """
    支持自动微分的变量类
    """
    
    def __init__(self, data: Union[float, int, np.ndarray], requires_grad: bool = False, grad_fn: Optional[Callable] = None):
        """
        初始化Variable
        
        Args:
            data: 数值数据 (标量或numpy数组)
            requires_grad: 是否需要梯度计算
            grad_fn: 梯度函数 (用于反向传播)
        """
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None
        
        # 用于构建计算图
        self.children = []  # 子节点 (输入变量)
        self.visited = False  # 反向传播访问标记
        
        # 如果需要梯度，初始化梯度为零
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
    
    def backward(self, grad_output: Optional[np.ndarray] = None):
        """
        反向传播计算梯度
        
        Args:
            grad_output: 从上游传来的梯度 (如果是标量输出，则为1)
        """
        if not self.requires_grad:
            return
        
        # 如果是标量输出且没有指定梯度，设为1
        if grad_output is None:
            grad_output = np.ones_like(self.data)
        
        # 累积梯度
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad_output
        
        # 如果有梯度函数，继续反向传播
        if self.grad_fn is not None:
            self.grad_fn(grad_output)
    
    def zero_grad(self):
        """清零梯度"""
        if self.grad is not None:
            self.grad.fill(0)
    
    def __repr__(self):
        return f"Variable(data={self.data}, requires_grad={self.requires_grad})"
    
    def __add__(self, other):
        """加法运算"""
        return add(self, other)
    
    def __radd__(self, other):
        """右加法运算"""
        return add(other, self)
    
    def __sub__(self, other):
        """减法运算"""
        return sub(self, other)
    
    def __rsub__(self, other):
        """右减法运算"""
        return sub(other, self)
    
    def __mul__(self, other):
        """乘法运算"""
        return mul(self, other)
    
    def __rmul__(self, other):
        """右乘法运算"""
        return mul(other, self)
    
    def __truediv__(self, other):
        """除法运算"""
        return div(self, other)
    
    def __rtruediv__(self, other):
        """右除法运算"""
        return div(other, self)
    
    def __pow__(self, other):
        """幂运算"""
        return power(self, other)
    
    def __neg__(self):
        """负号运算"""
        return neg(self)


def ensure_variable(x) -> Variable:
    """确保输入是Variable类型"""
    if not isinstance(x, Variable):
        return Variable(x)
    return x


# 基础数学运算函数

def add(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """加法运算"""
    a = ensure_variable(a)
    b = ensure_variable(b)
    
    result_data = a.data + b.data
    requires_grad = a.requires_grad or b.requires_grad
    
    def grad_fn(grad_output):
        if a.requires_grad:
            # 处理广播情况
            grad_a = grad_output
            if a.data.shape != grad_output.shape:
                # 对额外的维度求和
                ndims_added = grad_output.ndim - a.data.ndim
                for i in range(ndims_added):
                    grad_a = grad_a.sum(axis=0)
                # 对广播的维度求和
                for i, (dim_a, dim_grad) in enumerate(zip(a.data.shape, grad_a.shape)):
                    if dim_a == 1 and dim_grad > 1:
                        grad_a = grad_a.sum(axis=i, keepdims=True)
            a.backward(grad_a)
        
        if b.requires_grad:
            # 处理广播情况
            grad_b = grad_output
            if b.data.shape != grad_output.shape:
                # 对额外的维度求和
                ndims_added = grad_output.ndim - b.data.ndim
                for i in range(ndims_added):
                    grad_b = grad_b.sum(axis=0)
                # 对广播的维度求和
                for i, (dim_b, dim_grad) in enumerate(zip(b.data.shape, grad_b.shape)):
                    if dim_b == 1 and dim_grad > 1:
                        grad_b = grad_b.sum(axis=i, keepdims=True)
            b.backward(grad_b)
    
    result = Variable(result_data, requires_grad, grad_fn if requires_grad else None)
    result.children = [a, b] if requires_grad else []
    return result


def sub(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """减法运算"""
    a = ensure_variable(a)
    b = ensure_variable(b)
    
    result_data = a.data - b.data
    requires_grad = a.requires_grad or b.requires_grad
    
    def grad_fn(grad_output):
        if a.requires_grad:
            grad_a = grad_output
            if a.data.shape != grad_output.shape:
                ndims_added = grad_output.ndim - a.data.ndim
                for i in range(ndims_added):
                    grad_a = grad_a.sum(axis=0)
                for i, (dim_a, dim_grad) in enumerate(zip(a.data.shape, grad_a.shape)):
                    if dim_a == 1 and dim_grad > 1:
                        grad_a = grad_a.sum(axis=i, keepdims=True)
            a.backward(grad_a)
        
        if b.requires_grad:
            grad_b = -grad_output  # 减法的梯度是负的
            if b.data.shape != grad_output.shape:
                ndims_added = grad_output.ndim - b.data.ndim
                for i in range(ndims_added):
                    grad_b = grad_b.sum(axis=0)
                for i, (dim_b, dim_grad) in enumerate(zip(b.data.shape, grad_b.shape)):
                    if dim_b == 1 and dim_grad > 1:
                        grad_b = grad_b.sum(axis=i, keepdims=True)
            b.backward(grad_b)
    
    result = Variable(result_data, requires_grad, grad_fn if requires_grad else None)
    result.children = [a, b] if requires_grad else []
    return result


def mul(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """乘法运算"""
    a = ensure_variable(a)
    b = ensure_variable(b)
    
    result_data = a.data * b.data
    requires_grad = a.requires_grad or b.requires_grad
    
    def grad_fn(grad_output):
        if a.requires_grad:
            grad_a = grad_output * b.data
            if a.data.shape != grad_a.shape:
                ndims_added = grad_a.ndim - a.data.ndim
                for i in range(ndims_added):
                    grad_a = grad_a.sum(axis=0)
                for i, (dim_a, dim_grad) in enumerate(zip(a.data.shape, grad_a.shape)):
                    if dim_a == 1 and dim_grad > 1:
                        grad_a = grad_a.sum(axis=i, keepdims=True)
            a.backward(grad_a)
        
        if b.requires_grad:
            grad_b = grad_output * a.data
            if b.data.shape != grad_b.shape:
                ndims_added = grad_b.ndim - b.data.ndim
                for i in range(ndims_added):
                    grad_b = grad_b.sum(axis=0)
                for i, (dim_b, dim_grad) in enumerate(zip(b.data.shape, grad_b.shape)):
                    if dim_b == 1 and dim_grad > 1:
                        grad_b = grad_b.sum(axis=i, keepdims=True)
            b.backward(grad_b)
    
    result = Variable(result_data, requires_grad, grad_fn if requires_grad else None)
    result.children = [a, b] if requires_grad else []
    return result


def div(a: Union[Variable, float, int], b: Union[Variable, float, int]) -> Variable:
    """除法运算"""
    a = ensure_variable(a)
    b = ensure_variable(b)
    
    result_data = a.data / b.data
    requires_grad = a.requires_grad or b.requires_grad
    
    def grad_fn(grad_output):
        if a.requires_grad:
            grad_a = grad_output / b.data
            if a.data.shape != grad_a.shape:
                ndims_added = grad_a.ndim - a.data.ndim
                for i in range(ndims_added):
                    grad_a = grad_a.sum(axis=0)
                for i, (dim_a, dim_grad) in enumerate(zip(a.data.shape, grad_a.shape)):
                    if dim_a == 1 and dim_grad > 1:
                        grad_a = grad_a.sum(axis=i, keepdims=True)
            a.backward(grad_a)
        
        if b.requires_grad:
            grad_b = -grad_output * a.data / (b.data ** 2)
            if b.data.shape != grad_b.shape:
                ndims_added = grad_b.ndim - b.data.ndim
                for i in range(ndims_added):
                    grad_b = grad_b.sum(axis=0)
                for i, (dim_b, dim_grad) in enumerate(zip(b.data.shape, grad_b.shape)):
                    if dim_b == 1 and dim_grad > 1:
                        grad_b = grad_b.sum(axis=i, keepdims=True)
            b.backward(grad_b)
    
    result = Variable(result_data, requires_grad, grad_fn if requires_grad else None)
    result.children = [a, b] if requires_grad else []
    return result


def power(a: Variable, b: Union[Variable, float, int]) -> Variable:
    """幂运算"""
    a = ensure_variable(a)
    b = ensure_variable(b)
    
    result_data = np.power(a.data, b.data)
    requires_grad = a.requires_grad or b.requires_grad
    
    def grad_fn(grad_output):
        if a.requires_grad:
            # d/dx (x^n) = n * x^(n-1)
            grad_a = grad_output * b.data * np.power(a.data, b.data - 1)
            if a.data.shape != grad_a.shape:
                ndims_added = grad_a.ndim - a.data.ndim
                for i in range(ndims_added):
                    grad_a = grad_a.sum(axis=0)
                for i, (dim_a, dim_grad) in enumerate(zip(a.data.shape, grad_a.shape)):
                    if dim_a == 1 and dim_grad > 1:
                        grad_a = grad_a.sum(axis=i, keepdims=True)
            a.backward(grad_a)
        
        if b.requires_grad:
            # d/dx (a^x) = a^x * ln(a)
            grad_b = grad_output * result_data * np.log(a.data)
            if b.data.shape != grad_b.shape:
                ndims_added = grad_b.ndim - b.data.ndim
                for i in range(ndims_added):
                    grad_b = grad_b.sum(axis=0)
                for i, (dim_b, dim_grad) in enumerate(zip(b.data.shape, grad_b.shape)):
                    if dim_b == 1 and dim_grad > 1:
                        grad_b = grad_b.sum(axis=i, keepdims=True)
            b.backward(grad_b)
    
    result = Variable(result_data, requires_grad, grad_fn if requires_grad else None)
    result.children = [a, b] if requires_grad else []
    return result


def neg(a: Variable) -> Variable:
    """负号运算"""
    result_data = -a.data
    requires_grad = a.requires_grad
    
    def grad_fn(grad_output):
        if a.requires_grad:
            a.backward(-grad_output)
    
    result = Variable(result_data, requires_grad, grad_fn if requires_grad else None)
    result.children = [a] if requires_grad else []
    return result


def sum(a: Variable, axis=None, keepdims=False) -> Variable:
    """求和运算"""
    result_data = np.sum(a.data, axis=axis, keepdims=keepdims)
    requires_grad = a.requires_grad
    
    def grad_fn(grad_output):
        if a.requires_grad:
            # 梯度需要广播回原始形状
            grad_a = grad_output
            if not keepdims and axis is not None:
                # 如果没有保持维度，需要扩展维度
                if isinstance(axis, int):
                    grad_a = np.expand_dims(grad_a, axis)
                else:
                    for ax in sorted(axis):
                        grad_a = np.expand_dims(grad_a, ax)
            
            # 广播到原始形状
            grad_a = np.broadcast_to(grad_a, a.data.shape)
            a.backward(grad_a)
    
    result = Variable(result_data, requires_grad, grad_fn if requires_grad else None)
    result.children = [a] if requires_grad else []
    return result


def mean(a: Variable, axis=None, keepdims=False) -> Variable:
    """平均值运算"""
    result_data = np.mean(a.data, axis=axis, keepdims=keepdims)
    requires_grad = a.requires_grad
    
    def grad_fn(grad_output):
        if a.requires_grad:
            # 计算元素数量
            if axis is None:
                n = a.data.size
            else:
                if isinstance(axis, int):
                    n = a.data.shape[axis]
                else:
                    n = np.prod([a.data.shape[ax] for ax in axis])
            
            grad_a = grad_output / n
            if not keepdims and axis is not None:
                if isinstance(axis, int):
                    grad_a = np.expand_dims(grad_a, axis)
                else:
                    for ax in sorted(axis):
                        grad_a = np.expand_dims(grad_a, ax)
            
            grad_a = np.broadcast_to(grad_a, a.data.shape)
            a.backward(grad_a)
    
    result = Variable(result_data, requires_grad, grad_fn if requires_grad else None)
    result.children = [a] if requires_grad else []
    return result