main: main.o
	g++ -o $@ $< `pkg-config --libs opencv`

.cpp.o:
	g++ -o $@ -c $< `pkg-config --cflags opencv`
