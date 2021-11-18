OPENMP = 1
LIBSO = 1

OS := $(shell uname)

EXEC=otter
OBJDIR=./obj/

ifeq ($(LIBSO), 1)
LIBNAMESO = otter.so
endif

CC = g++ -std=c++11
OPTS = -Ofast
#CFLAGS = -Wall -Wfatal-errors -Wno-unused-result -Wno-unknown-pragmas -fPIC
CFLAGS = -fPIC

CFLAGS += $(OPTS)
LDFLAGS = -lm

ifeq ($(OPENMP), 1)
	ifeq ($(OS), Darwin)
		CFLAGS += -Xpreprocessor -fopenmp
		LDFLAGS += -lomp
	else
		CFLAGS += -fopenmp
	endif
endif

OBJ = main.o Neural_Network.o Otter.o Layer.o Tensor.o Utilities.o Box.o Data_Process.o Image_Process.o Jpeg.o Mat.o Mtcnn.o Test_All_Layer.o YOLOv3.o YOLOv4.o Machine_Learning.o

all: $(OBJDIR) backup $(EXEC) $(LIBNAMESO)

ifeq ($(LIBSO), 1)
CFLAGS+= -fPIC

$(LIBNAMESO): $(OBJDIR) $(OBJS)
	$(CC) -shared -std=c++11 -fvisibility=hidden $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard ./*.hpp)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)
backup:
	mkdir -p backup

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)
