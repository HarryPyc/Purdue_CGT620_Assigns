#pragma once
#include "cuda_runtime.h"

class Line {
public:
	float p[2], v[2];
	__host__ Line(float pu, float pv, float vu, float vv) {
		p[0] = pu, p[1] = pv;
		v[0] = vu, v[1] = vv;
	}
	__host__ ~Line() {}
	__host__ __host__ bool LineFunc(float i, float j) {
		float _v[2] = { i - p[0], j - p[1] };
		float val = _v[1] * v[0] - _v[0] * v[1];
		return val > 0.f;
	}
};