include ../../make.header

all: libimpacc.a

OBJ_DIR	= objs
LIB_DIR = ../lib
SOURCES	= $(wildcard *.cpp)
OBJS	= $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SOURCES))

libimpacc.a: $(OBJS) | $(LIB_DIR)
	ar -rcs $(LIB_DIR)/$@ $?
	cp $(LIB_DIR)/$@ ../../openarcrt

$(OBJS): $(OBJ_DIR)/%.o:%.cpp | $(OBJ_DIR)
	$(CXX) $(GOMP_FLAGS) $(MPI_FLAGS) $(GFRONTEND_INCLUDES) $(GBACKEND_INCLUDES) $< -c -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)
