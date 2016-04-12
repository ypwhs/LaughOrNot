# Paths
OPENCV_PATH=/usr/local

# Programs
CC=
CXX=g++

# Flags
ARCH_FLAGS=
CFLAGS=-Wextra -Wall -pedantic-errors $(ARCH_FLAGS) -O3 -Wno-long-long
LDFLAGS=$(ARCH_FLAGS)
DEFINES=
INCLUDES=-I$(OPENCV_PATH)/include -Iinclude/
LIBRARIES=-L$(OPENCV_PATH)/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lpthread

# Files which require compiling
SOURCE_FILES=\
	src/IO.cc\
	src/PDM.cc\
	src/Patch.cc\
	src/CLM.cc\
	src/FDet.cc\
	src/PAW.cc\
	src/FCheck.cc\
	src/Tracker.cc\
	src/Expression.cc\
	src/ExpressionClassifier.cc\


# Source files which contain a int main(..) function
SOURCE_FILES_WITH_MAIN=main.cc

# End Configuration
SOURCE_OBJECTS=$(patsubst %.cc,%.o,$(SOURCE_FILES))

ALL_OBJECTS=\
	$(SOURCE_OBJECTS) \
	$(patsubst %.cc,%.o,$(SOURCE_FILES_WITH_MAIN))

DEPENDENCY_FILES=\
	$(patsubst %.o,%.d,$(ALL_OBJECTS)) 

all: smile-detect

%.o: %.cc Makefile
	@# Make dependecy file
	$(CXX) -MM -MT $@ -MF $(patsubst %.cc,%.d,$<) $(CFLAGS) $(DEFINES) $(INCLUDES) $<
	@# Compile
	$(CXX) $(CFLAGS) $(DEFINES) -c -o $@ $< $(INCLUDES)

-include $(DEPENDENCY_FILES)

smile-detect: $(ALL_OBJECTS)
	$(CXX) $(LDFLAGS)  -o $@ $(ALL_OBJECTS) $(LIBRARIES)

.PHONY: clean
clean:
	@echo "Cleaning"
	@for pattern in '*~' '*.o' '*.d' ; do \
		find . -name "$$pattern" | xargs rm ; \
	done
