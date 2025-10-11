# C++ 自动微分框架 Makefile

CXX = g++
CXXFLAGS = -std=c++23 -Wall -Wextra -O2
TARGET1 = autograd_demo
TARGET2 = autograd_test
TARGET3 = test/mnist_cnn_demo
SOURCE1 = test/demo.cpp
SOURCE2 = test/test.cpp
SOURCE3 = test/mnist_cnn_demo.cpp
HEADER = autograd.hpp

# 默认目标
all: $(TARGET1) $(TARGET2) $(TARGET3)

# 编译主程序
$(TARGET1): $(SOURCE1) $(HEADER)
	$(CXX) $(CXXFLAGS) -o $(TARGET1) $(SOURCE1)

# 编译测试程序
$(TARGET2): $(SOURCE2) $(HEADER)
	$(CXX) $(CXXFLAGS) -o $(TARGET2) $(SOURCE2)

# 编译CNN演示程序
$(TARGET3): $(SOURCE3) $(HEADER)
	$(CXX) $(CXXFLAGS) -o $(TARGET3) $(SOURCE3)

# 运行主程序
run: $(TARGET1)
	./$(TARGET1)

# 运行测试程序
test: $(TARGET2)
	./$(TARGET2)

# 运行CNN演示程序
cnn: $(TARGET3)
	./$(TARGET3)

# 清理生成的文件
clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3)

# 显示帮助信息
help:
	@echo "可用的命令:"
	@echo "  make all    - 编译所有程序"
	@echo "  make run    - 编译并运行主演示程序"
	@echo "  make test   - 编译并运行测试程序"
	@echo "  make cnn    - 编译并运行CNN演示程序"
	@echo "  make clean  - 清理生成的可执行文件"
	@echo "  make help   - 显示此帮助信息"

.PHONY: all run test cnn clean help