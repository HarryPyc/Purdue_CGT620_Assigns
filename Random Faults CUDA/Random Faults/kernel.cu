/*
/*
 Random Faults in CUDA
 (C) Bedrich Benes 2020
 bbenes@purdue.edu
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <math.h>
#include <vector>			//Standard template library class
#include <GL/freeglut.h>

//in house created libraries
#include "vect3d.h"
#include "trackball.h"
#include "helper.h"       
#include "Line.cuh"


#pragma comment(lib, "freeglut.lib")

TrackBallC trackball;
bool mouseLeft, mouseMid, mouseRight;

GLint wWindow = 1200;
GLint hWindow = 800;


#define DEBUG
const int MAX = 256;
const int SCENE = 1;
const int maxSteps = 256;

GLint n = 1;
GLfloat a[MAX][MAX];
GLint fill = 1;
#define ELEV 0.0005f



//CUDA stuff
float *d_A;
Line* lines;

void Cleanup(bool noError)
{
	cudaError_t error;
	// Free device memory
	if (d_A) error = cudaFree(d_A);
	if (!noError || error != cudaSuccess) printf("Something failed \n");
}



void RandomFaultsCuda();

void Idle(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //clear all
	glEnable(GL_LIGHT0);
	trackball.Set3DViewCamera();
	glDisable(GL_LIGHTING);
	CoordSyst();
	glEnable(GL_LIGHTING);
	glCallList(SCENE);
	glutSwapBuffers();
}


Vect3d CrossProduct(Vect3d *a, Vect3d *b, Vect3d *cross)

{
	cross->SetX(a->GetY() * b->GetZ() - a->GetZ()* b->GetY());
	cross->SetY(a->GetZ() * b->GetX() - a->GetX()* b->GetZ());
	cross->SetZ(a->GetX() * b->GetY() - a->GetY()* b->GetX());
	return *cross;
}

void Display(void)
{
	int i, j;
	Vect3d v1, v2, v3, v13, v12, n;
	GLfloat materialColor[] = { 0.1f, 0.5f, 0.02f, 1.0f };
	GLfloat materialSpecular[] = { 0,0,0,1 };
	glNewList(SCENE, GL_COMPILE);
	glShadeModel(GL_SMOOTH);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, materialColor);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, materialSpecular);
	if (fill) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	else glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	for (i = 0; i < MAX - 1; i++)
	{
		glBegin(GL_TRIANGLE_STRIP);
		for (j = 0; j < MAX-1; j++)
		{
//triangle 1
			v1.Set(i / (GLfloat)MAX, j / (GLfloat)MAX, a[i][j]);
			v2.Set((i + 1) / (GLfloat)MAX, j / (GLfloat)MAX, a[i + 1][j]);
			v3.Set((i + 1) / (GLfloat)MAX, (j + 1) / (GLfloat)MAX, a[i + 1][j+1]);
			v12.Set(v1.GetX() - v3.GetX(), v1.GetY() - v3.GetY(), v1.GetZ() - v3.GetZ());
			v13.Set(v1.GetX() - v2.GetX(), v1.GetY() - v2.GetY(), v1.GetZ() - v2.GetZ());
			n.Set(CrossProduct(&v13,&v12,&n));
			n.Normalize();
			glNormal3fv(n);
			glVertex3fv(v1);
			glVertex3fv(v2);
			glVertex3fv(v3);
//triangle 2
			v1.Set(i / (GLfloat)MAX, j / (GLfloat)MAX, a[i][j]);
			v2.Set((i + 1) / (GLfloat)MAX, (j + 1) / (GLfloat)MAX, a[i + 1][j+1]);
			v3.Set((i) / (GLfloat)MAX, (j + 1) / (GLfloat)MAX, a[i][j+1]);
			v12.Set(v1.GetX() - v3.GetX(), v1.GetY() - v3.GetY(), v1.GetZ() - v3.GetZ());
			v13.Set(v1.GetX() - v2.GetX(), v1.GetY() - v2.GetY(), v1.GetZ() - v2.GetZ());
			n.Set(CrossProduct(&v13, &v12, &n));
			n.Normalize();
			glNormal3fv(n);
			glVertex3fv(v1);
			glVertex3fv(v2);
			glVertex3fv(v3);
		}
		glEnd();
	}
	glEndList();
}

void DisplayUgly(void)
{
	int i, j;

	glNewList(SCENE, GL_COMPILE);
	glColor3ub(0, 0, 0);
	if (fill) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	else glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	for (i = 0; i < MAX - 1; i++)
	{
		glBegin(GL_QUAD_STRIP);
		for (j = 0; j < MAX; j++)
		{
			glColor3f(a[i][j], a[i][j], a[i][j]);
			glVertex3f(i / (GLfloat)MAX, j / (GLfloat)MAX, a[i][j]);
			glVertex3f((i + 1) / (GLfloat)MAX, j / (GLfloat)MAX, a[i + 1][j]);
		}
		glEnd();
	}
	glEndList();
}

void Init(void)
{
	int i, j;

	glClearColor(1.0, 1.0, 1.0, 1.0);
	glClearDepth(1000.f);
	glEnable(GL_DEPTH_TEST);
	for (i = 0; i < MAX; i++)
		for (j = 0; j < MAX; j++)  a[i][j] = 0.5;

}


void myReshape(int w, int h)
{
	glViewport(0, 0, w, h);
	wWindow = w;
	hWindow = h;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-0.2, 1.2, -0.2, 1.2, -10, 10);
}


void RandomFault(void)
{
	//Write the CPU version here

}


void Key(unsigned char key, GLint i, GLint j)
{
	switch (key)
	{
	case 'f':
	case 'F': fill = (fill == 0); glutPostRedisplay(); break;
	case ' ': //run CPU implementation
	{
		long t1 = clock();
		for (int i = 0; i < maxSteps; i++)
		{
			char name[200];
			sprintf(name, "%i%% done\r", 100 * (i + 1) / maxSteps);
			glutSetWindowTitle(name);
			RandomFault();
		}
		long t2 = clock();
		glutSetWindowTitle("Random Faults in Cuda");
		printf("CPU Running time: %i\n", t2 - t1);
		break;
	}
	case 'c': //run CUDA implementation
	{
		glutSetWindowTitle("Running CUDA");
		long t1 = clock();
		RandomFaultsCuda();
		long t2 = clock();
		glutSetWindowTitle("Random Faults in Cuda");
		printf("CUDA Running time: %i\n", t2 - t1);
		break;
	}
	case 27:
	case 'q':
	case 'Q': exit(0);
	}
	glutPostRedisplay();
}

void Mouse(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		trackball.Set(true, x, y);
		mouseLeft = true;
	}
	if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
	{
		trackball.Set(false, x, y);
		mouseLeft = false;
	}
	if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN)
	{
		trackball.Set(true, x, y);
		mouseMid = true;
	}
	if (button == GLUT_MIDDLE_BUTTON && state == GLUT_UP)
	{
		trackball.Set(true, x, y);
		mouseMid = false;
	}
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
	{
		trackball.Set(true, x, y);
		mouseRight = true;
	}
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP)
	{
		trackball.Set(true, x, y);
		mouseRight = false;
	}
}

void MouseMotion(int x, int y) {
	if (mouseLeft)  trackball.Rotate(x, y);
	if (mouseMid)   trackball.Translate(x, y);
	if (mouseRight) trackball.Zoom(x, y);
//	glutPostRedisplay();
}




__global__ void RandFaultKernel(float a[MAX][MAX],  //2D array of elements
	const int N, //array is N*N
	const int n) //number of steps to run
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	if ((i>=N) || (j>=N)) return;
//Write the kernel here
	a[i][j] += (sin((float)i/ blockDim.x)+ cos((float)j / blockDim.y))*0.001;
}


void RandomFaultsCuda()
{
	cudaError_t error;
	int sizeArray;

	//allocate array on the device
	sizeArray = sizeof(float)*MAX*MAX; //2D array of floats
	error = cudaMalloc((void**)&d_A, sizeArray);
	//Copy the 2D array from host memory to device memory
	error = cudaMemcpy(d_A, a, sizeArray, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) Cleanup(false);

	//prepare blocks and grid
	const int BLOCKSIZE = 16;
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(ceil((float)MAX / dimBlock.x),
		         ceil((float)MAX / dimBlock.y));
	// Invoke kernel
	RandFaultKernel <<<dimGrid, dimBlock >>> ((float(*)[MAX])d_A, MAX, maxSteps);
	error = cudaGetLastError();
	if (error != cudaSuccess) printf("Something went wrong: %i\n", error);
	error = cudaThreadSynchronize();
	if (error != cudaSuccess) { printf("synchronization is wrong\n"); Cleanup(false); }
	// Copy result from device memory to host memory
	error = cudaMemcpy(a, d_A, sizeArray, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) { printf("could not copy from device\n"); Cleanup(false); }
	Cleanup(true);
}

// Host code
int main(int argc, char** argv)
{
	srand(5);
	glutInitWindowSize(wWindow, hWindow);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutCreateWindow("Random Faults");
	Init();
	glutDisplayFunc(Display);
	glutIdleFunc(Idle);
	glutKeyboardFunc(Key);
	glutReshapeFunc(myReshape);
	glutMouseFunc(Mouse);
	glutMotionFunc(MouseMotion);
	glutMainLoop();
	return 0;

}


