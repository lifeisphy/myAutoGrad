# C++ 自动微分框架 Makefile

CXX = g++
CXXFLAGS = -std=c++23 -Wall -Wextra -O2
TARGET1 = autograd_demo
TARGET2 = autograd_test
SOURCE1 = demo.cpp
SOURCE2 = test.cpp
HEADER = autograd.hpp

# 默认目标
all: $(TARGET1) $(TARGET2)

# 编译主程序
$(TARGET1): $(SOURCE1) $(HEADER)
	$(CXX) $(CXXFLAGS) -o $(TARGET1) $(SOURCE1)

# 编译测试程序
$(TARGET2): $(SOURCE2) $(HEADER)
	$(CXX) $(CXXFLAGS) -o $(TARGET2) $(SOURCE2)

# 运行主程序
run: $(TARGET1)
	./$(TARGET1)

# 运行测试程序
test: $(TARGET2)
	./$(TARGET2)

# 清理生成的文件
clean:
	rm -f $(TARGET1) $(TARGET2)

# 显示帮助信息
help:
	@echo "可用的命令:"
	@echo "  make all    - 编译所有程序"
	@echo "  make run    - 编译并运行主演示程序"
	@echo "  make test   - 编译并运行测试程序"
	@echo "  make clean  - 清理生成的可执行文件"
	@echo "  make help   - 显示此帮助信息"

.PHONY: all run test clean help