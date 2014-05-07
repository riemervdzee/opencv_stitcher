CC         = g++
CFLAGS     = -std=c++11 -o2 -c -Wall -fopenmp  `pkg-config opencv --cflags`
LDFLAGS    = -std=c++11 -o2 -fopenmp `pkg-config opencv --libs`
SOURCES    = main.cpp
OBJECTS    = obj/$(SOURCES:.cpp=.o)
EXECUTABLE = opencv_test

all: $(SOURCES) $(EXECUTABLE)


$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

obj/%.o: %.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm $(EXECUTABLE) $(OBJECTS)


