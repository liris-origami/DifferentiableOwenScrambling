#include <vector>
#include <random>

// #define WITH_OMP
#ifdef WITH_OMP
	#include <omp.h>
#endif

#include <iostream>
#include <cstring>
#include <algorithm>
#include <chrono>


namespace transport {

	// a 2d Vector class

	class Vector {
	public:
		explicit Vector(double x = 0, double y = 0) { coords[0] = x; coords[1] = y; }
		double operator[](int i) const {
			return coords[i];
		}
		double& operator[](int i) {
			return coords[i];
		}
		Vector& operator+=(const Vector& V) {
			coords[0] += V[0];
			coords[1] += V[1];
			return *this;
		}
		Vector& operator*=(double s) {
			coords[0] *= s;
			coords[1] *= s;
			return *this;
		}
		Vector& operator/=(double s) {
			coords[0] /= s;
			coords[1] /= s;
			return *this;
		}
		double getNorm2() const {
			return coords[0] * coords[0] + coords[1] * coords[1];
		}
		double getNorm() const {
			return sqrt(getNorm2());
		}
		void normalize() {
			double n = getNorm();
			coords[0] /= n;
			coords[1] /= n;
		}
		bool operator<(const Vector& b) const {
			if (coords[0] < b.coords[0] - 1E-12) return true;
			if (fabs(coords[0] - b.coords[0]) < 1E-12) {
				if (coords[1] < b.coords[1] - 1E-12) return true;
			}
			return false;
		}
		double coords[2];
	};
	/*bool operator==(const Vector& a, const Vector& b) {
		if (abs(a[0] - b[0]) > 1E-12) return false;
		if (abs(a[1] - b[1]) > 1E-12) return false;
		return true;
	}*/
	Vector operator*(double a, const Vector& B);
	Vector operator*(const Vector& B, double a);
	Vector operator*(const Vector& A, const Vector& B);
	Vector operator/(const Vector& B, double a);
	Vector operator/(const Vector& B, const Vector& A);
	Vector operator+(const Vector& A, const Vector& B);
	Vector operator-(const Vector& A, const Vector& B);
	Vector operator-(const Vector& A);
	double dot(const Vector& A, const Vector& B);

	template<typename T> T sqr(T x) {
		return x * x;
	}



	/// A Triangle with its neighbors as indices
	class Triangle {
	public:
		Triangle(int i, int j, int k, int n1, int n2, int n3) {
			id[0] = i;
			id[1] = j;
			id[2] = k;
			neigh[0] = n1;
			neigh[1] = n2;
			neigh[2] = n3;
		}
		int id[3];
		int neigh[3];
	};

	// A Triangle node in a Triangle mesh, stored as a linked list, with pointers to neighbors
	class TriangleP {
	public:
		TriangleP(int i, int j, int k, TriangleP* n1, TriangleP* n2, TriangleP* n3, bool infinite = false) {
			id[0] = i;
			id[1] = j;
			id[2] = k;
			neigh[0] = n1;
			neigh[1] = n2;
			neigh[2] = n3;
			infinite_visited_valid = 1; // 1*valid + 2*visited + 4*infinite
			if (infinite)
				infinite_visited_valid |= 4;
			nextInMesh = NULL;
			prevInMesh = NULL;
		}
		int id[3];
		TriangleP* neigh[3];

		TriangleP* nextInMesh;
		TriangleP* prevInMesh;
		unsigned char infinite_visited_valid; // stores 3 bools : whether the triangle is valid, whether it has been visited, and whether it is infinite
	};


	// "normal" orient 2d with 2x2 determinant. If within precision limits, just return 0
	int orient_2d_filter(const Vector& p0, const Vector& p1, const Vector p2);

	double integrateSquaredDistanceOver2DPixel(int pixX, int pixY, double imgW, const Vector& Pi);


	/// A Polygon. Great comment here.

	class Polygon {
	public:
		Polygon() {}

		std::vector<Vector> vertices;
		std::vector<int> neighbors;

		// https://www.seas.upenn.edu/~sys502/extra_materials/Polygon%20Area%20and%20Centroid.pdf

		double area() const {
			// https://wwwf.imperial.ac.uk/~rn/centroid.pdf
			double a = 0;
			size_t Nv = vertices.size();
			for (size_t i = 0; i < Nv; i++) {
				a += vertices[i][0] * vertices[(i + 1) % Nv][1] - vertices[(i + 1) % Nv][0] * vertices[i][1];
			}
			return fabs(a / 2.);
		}


		// quick test for whether a pixel (pixX, pixY) of an image of size imgW is fully inside (+1), outside (-1) or inbetween OR undetermined side (0) of the current convex polygon (assumes the image is square and represents the domain [0, 1]^2)
		int test_pixel_side(int pixX, int pixY, int imgW, const Vector& polygonCentroid, double inCircleRad, double outCircleRad) {

			double invW = 1 / (double)imgW;
			Vector centerP((pixX + 0.5) * invW, (pixY + 0.5) * invW);
			double db = (centerP - polygonCentroid).getNorm();
			if (db + 0.708 * invW < inCircleRad) return 1;
			if (db - 0.707 * invW > outCircleRad) return -1;
			//return 0;

			int nbPlus = 0; 
			int nbMinus = 0;
			const Vector P0(pixX * invW, pixY * invW);
			const Vector P2(P0[0] + invW, P0[1] + invW);

			const Vector P1(P2[0], P0[1]);
			const Vector P3(P0[0], P2[1]);

			for (int i = 0; i < vertices.size(); i++) {

				const Vector& vnext = vertices[(i + 1) % vertices.size()];
				char s1 = orient_2d_filter(vertices[i], vnext, P0);
				char s2 = orient_2d_filter(vertices[i], vnext, P1);
				char s3 = orient_2d_filter(vertices[i], vnext, P2);
				char s4 = orient_2d_filter(vertices[i], vnext, P3);

				if ((s1 | s2 | s3 | s4) == 0) return 0; // at least one is undetermined, we won't go further
				char s = s1 + s2 + s3 + s4;
				if (s == 4) nbPlus++;
				if (nbPlus > 1) return 0;

				if (s == -4) nbMinus++;				
			}
			if (nbPlus == 1) return -1; // it is definitely outside
			if (nbMinus == vertices.size()) return 1; // it is definitely inside
			return 0; // unknown or inbetween
		

		}

		// NOT the inscribed/circumscribed circles : just a circle of center "centroid" that is fully inside or fully contains the polygon
		// the polygon is convex => it contains its centroid
		void getInOutCircleRadii(const Vector& centroid, double& inCircleRad, double& outCircleRad) {

			double minD = 1E9, maxD = -1E9;
			int im = vertices.size() - 1;
			for (int i = 0; i < vertices.size(); i++) {
				Vector edge(vertices[i] - vertices[im]);
				Vector normalEdge(-edge[1], edge[0]);
				double dVert = (vertices[i] - centroid).getNorm2();
				double dEdge = dot(normalEdge, vertices[i] - centroid) / normalEdge.getNorm();
				minD = std::min(minD, dEdge);
				maxD = std::max(maxD, dVert);
				im = i;
			}

			inCircleRad = minD;
			outCircleRad = sqrt(maxD);
		}

		double weighted_area(double* density, int densityW, Vector& barycenter, bool integrate_square_dist = false, const Vector& Pi = Vector(0,0)) {
			 
			if (vertices.size() <= 2) return 0.;

			double val = 0;
			double bminx = 1E9, bminy = 1E9, bmaxx = -1E9, bmaxy = -1E9;
			for (int j = 0; j < vertices.size(); j++) {
				bminx = std::min(bminx, vertices[j][0]);
				bminy = std::min(bminy, vertices[j][1]);
				bmaxx = std::max(bmaxx, vertices[j][0]);
				bmaxy = std::max(bmaxy, vertices[j][1]);
			}
			bminx = std::min(densityW - 1., std::max(0., std::floor(densityW * bminx)));
			bminy = std::min(densityW - 1., std::max(0., std::floor(densityW * bminy)));
			bmaxx = std::min(densityW - 1., std::max(0., std::floor(densityW * bmaxx)));
			bmaxy = std::min(densityW - 1., std::max(0., std::floor(densityW * bmaxy)));

			
			Vector centroid_uniform = this->centroid(); 
			double inCircleRad, outCircleRad;
			this->getInOutCircleRadii(centroid_uniform, inCircleRad, outCircleRad);
			//int nb0=0, nb1 = 0;
			barycenter = Vector(0, 0);
			Polygon vor, tmp;
			vor.vertices.reserve(6);
			tmp.vertices.reserve(6);
			for (int y = bminy; y <= bmaxy; y++) {
				for (int x = bminx; x <= bmaxx; x++) {

										
					int s = this->test_pixel_side(x, y, densityW, centroid_uniform, inCircleRad, outCircleRad); // right now, this filter doesn't speed up anything
					//nb1 += abs(s);
					//nb0 += 1 - abs(s);
					if (s == -1) continue;
					if (s == 1) {
						if (integrate_square_dist) {
							val += integrateSquaredDistanceOver2DPixel(x, y, densityW, Pi) * density[y * densityW + x];
						} else {
						    double a = density[y * densityW + x] / (double)(densityW * densityW);
							val += a;
							barycenter += a * Vector(x+0.5, y+0.5)/densityW;
						}
						continue;
					}

					this->intersect_with_pixel(x, y, densityW, vor, tmp);					
					if (integrate_square_dist) {
						val += vor.integrateSquaredDistance2D(Pi)* density[y * densityW + x];
					} else {
						double a = vor.area() * density[y * densityW + x];
						val += a ;
						barycenter += a * vor.centroid();
					}

				}
			}
			//std::cout << nb0 << " " << nb1 << std::endl;
			barycenter = barycenter / val;

			return val ;

		} 

		Vector centroid() const {
			double A = 0;
			Vector C(0., 0.);
			size_t Nv = vertices.size();
			if (Nv == 0) return Vector(0., 0.);
			if (Nv == 1) return vertices[0];
			if (Nv == 2) return (vertices[0] + vertices[1]) * 0.5;
			Vector avg(0., 0.);
			for (size_t i = 0; i < Nv; i++) {
				double cc = vertices[i][0] * vertices[(i + 1) % Nv][1] - vertices[(i + 1) % Nv][0] * vertices[i][1];
				A += cc;
				avg += vertices[i];
				for (int k = 0; k < 2; k++) {
					C[k] += cc * (vertices[i][k] + vertices[(i + 1) % Nv][k]);
				}
			}
			A *= 0.5;
			if (A == 0) return avg / (double)Nv;
			Vector c = C / (6 * A);
			return c;
		}

		double integrateSquaredDistance2D(const Vector& Pi) { //https://people.sc.fsu.edu/~jburkardt/cpp_src/polygon_integrals/polygon_integrals.html
			double s = 0;
			size_t Nv = vertices.size();
			double npi2 = Pi.getNorm2();
			size_t im = Nv - 1;
			for (size_t i = 0; i < Nv; i++) {
				double sm = vertices[im].getNorm2() + vertices[i].getNorm2() + dot(vertices[im], vertices[i]) - 4 * dot(Pi, vertices[im] + vertices[i]) + 6 * npi2;
				s += (vertices[im][0] * vertices[i][1] - vertices[i][0] * vertices[im][1]) * sm;
				im = i;
			}
			double v1 = fabs(s) / 12.;
			return v1;
		}


		void clip_polygon_by_edge(const Vector& u, const Vector& v, Polygon& result) const { 
			
			result.vertices.clear();
			result.vertices.reserve(vertices.size() + 1);
			Vector N(v[1] - u[1], -v[0] + u[0]);

			for (int i = 0; i < vertices.size(); i++) {

				const Vector& A = (i == 0) ? vertices[vertices.size() - 1] : vertices[i - 1];
				const Vector& B = vertices[i];

				if (dot(B - u, N) < 0) { // B is inside

					if (dot(A - u, N) >= 0) { // A is outside
						double t = dot(u - A, N) / dot(B - A, N);
						Vector P = A + t * (B - A);
						result.vertices.push_back(P);
					}
					result.vertices.push_back(B);
				} else {
					if (dot(A - u, N) < 0) { // A is inside
						double t = dot(u - A, N) / dot(B - A, N);
						Vector P = A + t * (B - A);
						result.vertices.push_back(P);
					}
				}
			}
		}

		void intersect_with_pixel(int px, int py, int width, Polygon& result, Polygon &tmp) const {


			Vector p0(px / (double)width, py / (double)width);
			Vector p3(px / (double)width, (py+1) / (double)width);
			Vector p2((px + 1) / (double)width, (py + 1) / (double)width);
			Vector p1((px + 1) / (double)width, py / (double)width);

			this->clip_polygon_by_edge(p0, p1, tmp);  if (tmp.vertices.size() == 0) { result = tmp;  return; };
			tmp.clip_polygon_by_edge(p1, p2, result); if (result.vertices.size() == 0) return;
			result.clip_polygon_by_edge(p2, p3, tmp);
			tmp.clip_polygon_by_edge(p3, p0, result);
		}

	};


	// not used anymore since using Shewchuk's way to checking inCircle. Could be useful anyway.
	Vector circumcenter(const Vector& A, const Vector& B, const Vector& C);

	double det22(const Vector& A, const Vector& B);

	// https://www.cs.cmu.edu/~quake/robust.html
	bool isInCircle(const Vector& P, const Vector& A, const Vector& B, const Vector& C);


	// https://www.cs.cmu.edu/~quake/robust.html but with weights and without the "with a little more effort" transform. height is the parabolic lifting with weight: x^2+y^2+w^2
	bool isInCircleLifted(const Vector& P, const Vector& A, const Vector& B, const Vector& C, double hP, double hA, double hB, double hC);


	int getQuadrant(const Vector& p, double xMid, double yMid);

	// Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates, JR Shewchuk DCG 1997
	// Use expansion arithmetic to produce robust orientation predicates (not used for inCircle as it seems already much more robust)
	// basic idea: if some operation is within precision limits, convert it to an approximate value and a remainder such that the sum of both is still the exact value.


	// arithmetic presented in Shewchuk's paper
	void fast_two_sum(double a, double b, double& x, double& y);
	void fast_two_diff(double a, double b, double& x, double& y);
	void two_sum(double a, double b, double& x, double& y);
	void two_diff(double a, double b, double& x, double& y);

	void split(double a, double& ahi, double& alo);
	void two_product(double a, double b, double& x, double& y);



	// Stores an Expansion, i.e., a series of terms such that their sum is the exact value. 
	// For side predicates in 2d, 8 values at max to store 
	// (compression not implemented)

	template<int DMAX = 8>
	class Expansion {
	public:
		Expansion() { nex = 0; };

		template<int DMAX2>
		void fromExp(const Expansion<DMAX2>& b) {
			memcpy(ex, b.ex, b.nex * sizeof(double));
			nex = b.nex;
		}
		void create_from_sum(double a, double b) {
			nex = 2;
			two_sum(a, b, ex[0], ex[1]);
		}
		void create_from_diff(double a, double b) {
			nex = 2;
			two_diff(a, b, ex[0], ex[1]);
		}
		void grow_add(double b) {
			double q = b;
			for (int i = 0; i < nex; i++) {
				two_sum(q, ex[i], q, ex[i]);
			}
			ex[nex] = q;
			nex++;
		}
		template<int DMAX2>
		void sum(const Expansion<DMAX2>& f) {
			// assert nex+f.nex<8
			for (int i = 0; i < f.nex; i++) {
				grow_add(f.ex[i]);
			}
		}
		void expansion_diff(double x1, double y1, double x2, double y2) {
			ex[0] = x1;
			ex[1] = y1;
			nex = 2;
			grow_add(-x2);
			grow_add(-y2);
		}
		int sign() {
			if (nex == 0) return 0;
			bool iszero = true;
			for (int i = 0; i < nex; i++) {
				if (ex[i] != 0) {
					iszero = false;
					break;
				}
			}
			if (iszero) return 0;
			if (ex[nex - 1] > 0) return 1; else return -1;

		}
		int nex;
		double ex[DMAX];
	};


	// Fig 21 of Shewchuk's paper
	// Computes the determinant exactly
	int orient_exact(double ax, double ay, double bx, double by, double cx, double cy);

	// the actual 2D orientation code : first try the approximate way, and if it fails, the exact (but much slower) predicate
	int orient(const Vector& A, const Vector& B, const Vector C);



	// the 2 functions below are by ChatGPT 4	
	int64_t doubleToRawInt(double value); // Convert a normalized double (in range [0, 1)) to its 64-bit integer representation
	bool less_msb(int64_t x, int64_t y);


	class Compare {
	public:
		Compare(const std::vector<Vector>& vertices, int axis) : vertices(vertices), axis(axis) {}

		bool operator()(int i, int j) {
			return vertices[i][axis] < vertices[j][axis];
		}

	private:
		const std::vector<Vector>& vertices;
		int axis;
	};

	// The Bowyer Watson algorithm to compute a Delaunay, and its dual
	class Bowyer2D {
	public:
		Bowyer2D() {
		};


		// find the triangle containing v, starting from the last inserted triangle
		// we start from the last inserted triangle since vertices are stored in a Morton curve order so it shouldn't be too far away
		TriangleP* locateTriangleP(const Vector& v) {

			TriangleP* t = lastvP;

			int i;
			do {

				if (t->infinite_visited_valid & 4) break; // infinite triangle

				for (i = 0; i < 3; i++) {
					if (orient(v, sorted_vertices[t->id[i]], sorted_vertices[t->id[(i + 1) % 3]]) < 0) { // walk along finite triangles
						t = t->neigh[i];
						break;
					}
				}

			} while (i != 3);

			return t;
		}



		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////// This part of the code below contains helpers for ordering vertices along a Morton curve and Hilbert curves                          //////////////////////// 
		////////  It has been generated with ChatGPT 4 (no other part of the code is AI, apart from doubleToRawInt / less_msb)                       //////////////////////// 
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		bool cmp_zorder(size_t lhs, size_t rhs) {

			size_t msd = 0;
			for (size_t dim = 1; dim < 2; ++dim) {
				int64_t lhsInt = doubleToRawInt(vertices[lhs][msd]);
				int64_t rhsInt = doubleToRawInt(vertices[rhs][msd]);

				int64_t lhsDimInt = doubleToRawInt(vertices[lhs][dim]);
				int64_t rhsDimInt = doubleToRawInt(vertices[rhs][dim]);

				if (less_msb(lhsInt ^ rhsInt, lhsDimInt ^ rhsDimInt)) {
					msd = dim;
				}
			}
			return vertices[lhs][msd] < vertices[rhs][msd];
		}

		int hilbertSplit(std::vector<int>& permutation, int begin, int end, bool useX, bool invert) {
			int middle = begin + (end - begin) / 2;

			if (useX) {
				std::nth_element(permutation.begin() + begin, permutation.begin() + middle, permutation.begin() + end,
					[this, invert](int a, int b) { return invert ? vertices[a][0] > vertices[b][0] : vertices[a][0] < vertices[b][0]; });
			} else {
				std::nth_element(permutation.begin() + begin, permutation.begin() + middle, permutation.begin() + end,
					[this, invert](int a, int b) { return invert ? vertices[a][1] > vertices[b][1] : vertices[a][1] < vertices[b][1]; });
			}

			return middle;
		}

		// https://doc.cgal.org/latest/Spatial_sorting/index.html
		void hilbertSort(std::vector<int>& permutation, int begin, int end, bool useXFirst = true, bool invertFirst = false) {
			if (end - begin <= 1) return;

			int middle2 = hilbertSplit(permutation, begin, end, useXFirst, invertFirst);
			int middle1 = hilbertSplit(permutation, begin, middle2, !useXFirst, !invertFirst);
			int middle3 = hilbertSplit(permutation, middle2, end, !useXFirst, invertFirst);

			hilbertSort(permutation, begin, middle1, !useXFirst, !invertFirst);
			hilbertSort(permutation, middle1, middle2, useXFirst, invertFirst);
			hilbertSort(permutation, middle2, middle3, useXFirst, invertFirst);
			hilbertSort(permutation, middle3, end, !useXFirst, invertFirst);
		}


		//////////////////////////// end of ChatGPT Code /////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////



		// by default sort according to Hilbert curve ; otherwise Z curve
		void sort_vertices(bool hilbert = true) {

			lifted = weights.size() != 0;

			perm.resize(vertices.size());
			for (int i = 0; i < vertices.size(); i++) {
				perm[i] = i;
			}
			int maxDepth = std::ceil(std::log2(std::sqrt(perm.size())));
			if (hilbert)
				hilbertSort(perm, 0, perm.size());
			else
				std::sort(perm.begin(), perm.end(), [this](int a, int b) {return this->cmp_zorder(a, b); });
			

			sorted_vertices.resize(vertices.size());
			sorted_weights.resize(vertices.size());

			if (lifted) {
				std::transform(perm.begin(), perm.end(), sorted_weights.begin(), [&](std::size_t i) { return weights[i]; }); // sort the height according to the sorting permutation perm
			}
			std::transform(perm.begin(), perm.end(), sorted_vertices.begin(), [&](std::size_t i) { return vertices[i]; }); // sort the vertices according to the sorting permutation perm		
		}


		std::vector<int> perm; // permutation for ordering vertices along a morton curve.


		// vertices are sorted by default. if need_sorting == false, this assumes the "perm" vector has been already properly initialized (e.g., by a previous call to compute_delaunay(true)).
		void compute_delaunay(bool need_sorting = true) {

			lifted = weights.size() != 0;

			// sort vertices with Z ordering ; since we need to order the weights in the same way, we sort a permutation array and apply it to both
			// https://stackoverflow.com/questions/17074324/how-can-i-sort-two-vectors-in-the-same-way-with-criteria-that-uses-only-one-of

			if (need_sorting) {
				sort_vertices();
			} else {

				// the permutation array perm has already been computed before, just apply it
				sorted_vertices.resize(vertices.size());
				sorted_weights.resize(vertices.size());

				if (lifted) {
					std::transform(perm.begin(), perm.end(), sorted_weights.begin(), [&](std::size_t i) { return weights[i]; }); // sort the height according to the sorting permutation perm
				}
				std::transform(perm.begin(), perm.end(), sorted_vertices.begin(), [&](std::size_t i) { return vertices[i]; }); // sort the vertices according to the sorting permutation perm
			}


			if (lifted) {

				// compute the parabolic lifting, where h[i] = sqrt(M - weights[i]) and we use x^2 + y^2 + h^2 (where M doesn't matter). This is used in the inCircle predicate only, for weighted triangulations
				sorted_heights.resize(sorted_vertices.size());
				for (int i = 0; i < sorted_vertices.size(); i++) {
					sorted_heights[i] = sorted_vertices[i].getNorm2() - sorted_weights[i];
				}
			}

			// add a virtual vertex at infinity, to be used for correct handling of the convex hull
			// old answer by Bruno Levy here: https://stackoverflow.com/questions/3642548/infinite-initial-bounding-triangle-in-iterative-delaunay-triangulators?ref=gorillasun.de
			sorted_vertices.push_back(Vector(INFINITY, INFINITY));

			// let's use the first 3 vertices as our initial triangle, but correctly oriented in CCW order
			if (orient(sorted_vertices[0], sorted_vertices[1], sorted_vertices[2]) < 0) {
				std::swap(sorted_vertices[1], sorted_vertices[2]);
				std::swap(perm[1], perm[2]);
				if (lifted) {
					std::swap(sorted_weights[1], sorted_weights[2]);
					std::swap(sorted_heights[1], sorted_heights[2]);
				}
			}

			// add virtual vertex at infinity, and connect the initial triangle (initTri) to it via its 3 sides

			TriangleP* adj1, * adj2, * adj3;

			adj1 = new TriangleP(1, 0, sorted_vertices.size() - 1, NULL, NULL, NULL, true);
			adj1->prevInMesh = NULL;

			adj2 = new TriangleP(2, 1, sorted_vertices.size() - 1, NULL, NULL, NULL, true);
			adj1->nextInMesh = adj2;
			adj2->prevInMesh = adj1;

			adj3 = new TriangleP(0, 2, sorted_vertices.size() - 1, NULL, NULL, NULL, true);
			adj2->nextInMesh = adj3;
			adj3->prevInMesh = adj2;

			TriangleP* initTri = new TriangleP(0, 1, 2, adj1, adj2, adj3);
			initTri->nextInMesh = NULL;
			initTri->prevInMesh = adj3;
			adj3->nextInMesh = initTri;

			TriangleP* triangulationP = adj1; //our triangulation, stored as a linked list, is triangulationP

			triangulationP->neigh[0] = adj1;
			triangulationP->neigh[1] = adj2;
			triangulationP->neigh[2] = adj3;

			adj1->neigh[0] = initTri;
			adj2->neigh[0] = initTri;
			adj3->neigh[0] = initTri;


			adj1->neigh[1] = adj3;
			adj2->neigh[1] = adj1;
			adj3->neigh[1] = adj2;

			adj1->neigh[2] = adj2;
			adj2->neigh[2] = adj3;
			adj3->neigh[2] = adj1;


			lastvP = initTri;
			TriangleP* firstInList = triangulationP;
			TriangleP* lastInList = lastvP;

			// these could be any structure that provides a stack for visiting a graph (a linked list, ...) but for a depth first traversal, we only need pushback and popback so we can just use an array
			// avoid reallocations by using a fixed-size array. Hopefully big enough for big problems.
			TriangleP* potentiallybad[5000];
			TriangleP* visitedlist[5000];
			TriangleP* bad[5000];

			int nbpotentiallybad = 0; // just stores the actual array size (might be similar in perf to an std::vector with big .reserve memory)
			int nbvisited = 0;
			int nbbad = 0;


			// start inserting vertices one by one to the triangulation

			for (int i = 3; i < sorted_vertices.size() - 1; i++) { // we add all samples (except the first 3 ones already in there, and the last infinite vertex)


				// for each sample (or vertex of the Delaunay graph) we insert it in the current Delaunay
				// this means: first finding in which triangle of the current Delaunay it belongs, by walking along the triangulation from the last inserted triangle
				// and then checking whether it is in conflict with this triangle (maybe not for weighted Delaunay) and if so, if it's is conflict with its neighboring triangles (and propagating recursively)
				// Once the conflicted list of triangles is found, we remove them and stellate the hole (i.e., draw an edge between the inserted vertex and all vertices on the border of this hole)

				Vector v = sorted_vertices[i];
				TriangleP* ti = locateTriangleP(v);

				if (lifted && !isInCircleP(i, ti)) continue; // dangling vertex : this can happen for weighted delaunay only (the corresponding power cell is empty)
				
				// we clear the lists
				nbpotentiallybad = 0;
				nbvisited = 0;
				nbbad = 0;


				// recursively visit neighbors and find which triangles are conflicted (not respecting the inCircle predicate)

				potentiallybad[nbpotentiallybad] = ti; nbpotentiallybad++;   // ==> potentiallybad.push_back(ti);

				ti->infinite_visited_valid |= 2;                             // ==>  ti->visited = true;
				visitedlist[nbvisited] = ti; nbvisited++;                    // ==> visitedlist.push_back(ti);

				while (nbpotentiallybad > 0) {
					TriangleP* cur = potentiallybad[nbpotentiallybad - 1];  //  ==> potentiallybad.back();
					nbpotentiallybad--;                                     //  ==> potentiallybad.pop_back();

					if (isInCircleP(i, cur)) {

						bad[nbbad] = cur; nbbad++;                          //  ==> bad.push_back(cur);					
						cur->infinite_visited_valid &= ~1;                  //  ==> cur->valid = false;

						for (int j = 0; j < 3; j++) {
							//if (cur->neigh[j] == NULL) continue;            // should not happen since we have the infinite vertex now.

							if (!(cur->neigh[j]->infinite_visited_valid & 2)) {  //  ==>  if (!cur->neigh[j]->visited) {

								potentiallybad[nbpotentiallybad] = cur->neigh[j]; nbpotentiallybad++;  //  ==>  potentiallybad.push_back(cur->neigh[j]);

								cur->neigh[j]->infinite_visited_valid |= 2;                            // ==>  cur->neigh[j]->visited = true;
								visitedlist[nbvisited] = cur->neigh[j]; nbvisited++;                   //  ==> visitedlist.push_back(cur->neigh[j]);				
							}
						}
					}
				}				

				// we reset the state of visited triangles for the next round.
				for (int j = 0; j < nbvisited; j++) {
					visitedlist[j]->infinite_visited_valid &= ~2;  // ==> visitedlist[j]->visited = false;
				}

				// find the first (or even one) triangle on the border of the conflicted region
				TriangleP* firstOnBorder = NULL;
				int firstBorderEdge = -1;
				for (int badid = 0; badid < nbbad; badid++) {
					bool found = false;
					for (int j = 0; j < 3; j++) {
						if (bad[badid]->neigh[j]->infinite_visited_valid & 1) {

							firstOnBorder = bad[badid];
							firstBorderEdge = j;
							found = true;
							break;
						}
					}
					if (found) break;
				}


				// We now traverse all edges on the border and connect border vertices to the new inserted vertex (so we add new triangles)

				TriangleP* curBorderTriangle = firstOnBorder;
				int curBorderEdge = firstBorderEdge;
				int firstBorderVtx = firstOnBorder->id[firstBorderEdge];

				TriangleP* previouslyInsertedTriangle = NULL;
				TriangleP* firstInsertedTriangle = NULL;
				do {
					// add Triangle with appropriate neighbor information
					lastInList->nextInMesh = new TriangleP(curBorderTriangle->id[curBorderEdge], curBorderTriangle->id[(curBorderEdge + 1) % 3], i, curBorderTriangle->neigh[curBorderEdge], NULL, previouslyInsertedTriangle);

					// this may be an infinite triangle (touching the inserted infinite vertex at the end) ; flag it as such
					if (curBorderTriangle->id[curBorderEdge] == sorted_vertices.size() - 1 || curBorderTriangle->id[(curBorderEdge + 1) % 3] == sorted_vertices.size() - 1)
						lastInList->nextInMesh->infinite_visited_valid |= 4;

					// keep track of pointers
					if (previouslyInsertedTriangle)
						previouslyInsertedTriangle->neigh[1] = lastInList->nextInMesh;
					else
						firstInsertedTriangle = lastInList->nextInMesh;

					// update neighborhood information of its neighbors
					if (curBorderTriangle->neigh[curBorderEdge]) {
						for (int j = 0; j < 3; j++) {
							if (curBorderTriangle->neigh[curBorderEdge]->neigh[j] == curBorderTriangle) {
								curBorderTriangle->neigh[curBorderEdge]->neigh[j] = lastInList->nextInMesh;
								break;
							}
						}
					}
					previouslyInsertedTriangle = lastInList->nextInMesh;
					lastInList->nextInMesh->prevInMesh = lastInList;
					lastInList->nextInMesh->nextInMesh = NULL;
					lastInList = lastInList->nextInMesh;



					// crappy code to find the next edge on the boundary... :
					// first try another edge of the same triangle

					int nexEdge = (curBorderEdge + 1) % 3;
					if (curBorderTriangle->neigh[nexEdge]->infinite_visited_valid & 1) {
						curBorderEdge = nexEdge;
					} else { 				// if it was not an edge on the boundary (i.e., adjacent to a valid triangle), check an adjacent triangle

						int pivotEdge = (curBorderEdge + 1) % 3;

						do {
							TriangleP* prevBorderTriangle = curBorderTriangle;
							for (int j = 0; j < 3; j++) {
								if (curBorderTriangle->neigh[pivotEdge] && curBorderTriangle->neigh[pivotEdge]->neigh[j] == curBorderTriangle) {
									curBorderTriangle = curBorderTriangle->neigh[pivotEdge];
									pivotEdge = (j + 1) % 3;
									break;
								}
							}
							curBorderEdge = pivotEdge;
						} while (curBorderTriangle->neigh[curBorderEdge] && !(curBorderTriangle->neigh[curBorderEdge]->infinite_visited_valid & 1)   /* ==> !curBorderTriangle->neigh[curBorderEdge]->valid*/);

					}


				} while (firstBorderVtx != curBorderTriangle->id[(curBorderEdge) % 3]); // while we have not returned back to where we started

				lastInList->neigh[1] = firstInsertedTriangle;
				firstInsertedTriangle->neigh[2] = lastInList;

				// physically kill the bad triangles from the mesh
				for (int badid = 0; badid < nbbad; badid++) {
					TriangleP* curP = bad[badid];
					if (curP == firstInList) {
						firstInList = firstInList->nextInMesh;
						triangulationP = firstInList;
					}
					if (curP == lastInList) lastInList = lastInList->prevInMesh;
					if (curP->prevInMesh) curP->prevInMesh->nextInMesh = curP->nextInMesh;
					if (curP->nextInMesh) curP->nextInMesh->prevInMesh = curP->prevInMesh;
					delete curP; // pan pan.
				}

				// the hint for starting the next "locate" is the last inserted triangle
				lastvP = firstInsertedTriangle;

				// unless it is infinite ; in this case use the one inserted before (could use a neighbor instead).
				while (lastvP->infinite_visited_valid & 4) {
					lastvP = lastvP->nextInMesh;
				}
			}


			// convert our crappy linked list to a nice triangle soup, and we don't care about adjacencies.
			// number of triangles = nb of vertices*2 - 2 -nb boundary vertices
			// http://www.cs.uu.nl/geobook/interpolation.pdf Th 9.1, here with at least 3 vertices on the boundary
			triangles.clear();
			triangles.reserve(sorted_vertices.size() * 2 - 5);

			TriangleP* curP = firstInList;
			while (curP) {
				if (curP->id[0] == sorted_vertices.size() - 1 || curP->id[1] == sorted_vertices.size() - 1 || curP->id[2] == sorted_vertices.size() - 1) {
					curP = curP->nextInMesh;
					if (curP) delete curP->prevInMesh;
					continue;
				}

				triangles.push_back(Triangle(perm[curP->id[0]], perm[curP->id[1]], perm[curP->id[2]], -1, -1, -1));
				curP = curP->nextInMesh;

				if (curP)
					delete curP->prevInMesh;
			}

		}

		// checks if P==vertices[pointID] is within triangle t circumcircle
		// for infinite triangles, see https://stackoverflow.com/questions/3642548/infinite-initial-bounding-triangle-in-iterative-delaunay-triangulators?ref=gorillasun.de
		bool isInCircleP(int pointID, TriangleP* t) {

			const Vector& P = sorted_vertices[pointID];

			if (t->infinite_visited_valid & 4) { // infinite triangle

				int idInf = -1;
				for (int i = 0; i < 3; i++) {
					if (t->id[i] == sorted_vertices.size() - 1) {
						idInf = i;
						break;
					}
				}
				int orientation = orient(P, sorted_vertices[t->id[(idInf + 1) % 3]], sorted_vertices[t->id[(idInf + 2) % 3]]); //not working for exact predicates

				if (orientation > 0) return true;
				if (orientation < 0) return false;

				// if point exactly in the middle, check opposite triangle: 
				return isInCircleP(pointID, t->neigh[(idInf + 1) % 3]);
			} else
				// standard case (for weighted an unweighted delaunay)
				if (lifted) {
					return transport::isInCircleLifted(P, sorted_vertices[t->id[0]], sorted_vertices[t->id[1]], sorted_vertices[t->id[2]], sorted_heights[pointID], sorted_heights[t->id[0]], sorted_heights[t->id[1]], sorted_heights[t->id[2]]);
				} else {
					return transport::isInCircle(P, sorted_vertices[t->id[0]], sorted_vertices[t->id[1]], sorted_vertices[t->id[2]]);
				}
		}

		// save an svg file showing the delaunay, power diagram and vertices (all optionals)
		void save_svg(std::string filename, bool save_triangulation = true, bool save_voronoi = true, bool save_samples = true, bool save_ordering = false) {

			FILE* f = fopen(filename.c_str(), "w+");
			fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"4000\" height = \"4000\">\n");

			if (voronoi.size() != 0 && save_voronoi) {
				fprintf(f, "<g>\n");
				for (int i = 0; i < voronoi.size(); i++) {
					fprintf(f, "<polygon points = \"");
					for (int j = 0; j < voronoi[i].vertices.size(); j++) {
						fprintf(f, "%3.3f, %3.3f ", (voronoi[i].vertices[j][0] * 4000), (4000 - voronoi[i].vertices[j][1] * 4000));
					}
					fprintf(f, "\"\nfill = \"#005BBB\" stroke = \"black\"/>\n");
				}
				fprintf(f, "</g>\n");
			}


			if (triangles.size() != 0 && save_triangulation) {
				fprintf(f, "<g>\n");
				for (int i = 0; i < triangles.size(); i++) {
					fprintf(f, "<polygon points = \"");
					for (int j = 0; j < 3; j++) {
						fprintf(f, "%u, %u ", (int)(vertices[triangles[i].id[j]][0] * 4000), (int)(4000 - vertices[triangles[i].id[j]][1] * 4000));
					}
					fprintf(f, "\"\nfill = \"none\" stroke = \"orange\" stroke-width=\"1\" />\n");
				}
				fprintf(f, "</g>\n");
			}

			if (sorted_vertices.size() != 0 && save_ordering) {
				fprintf(f, "<g>\n");
				for (int i = 0; i < sorted_vertices.size()-1; i++) {
					fprintf(f, "<line x1=\"%u\" y1=\"%u\" x2=\"%u\" y2=\"%u\" stroke=\"black\" stroke-width=\"1\"  />", (int)(sorted_vertices[i][0] * 4000), (int)(4000 - sorted_vertices[i][1] * 4000), (int)(sorted_vertices[i+1][0] * 4000), (int)(4000 - sorted_vertices[i+1][1] * 4000));
				}
				fprintf(f, "</g>\n");
			}

			if (vertices.size() != 0 && save_samples) {
				fprintf(f, "<g>\n");
				for (int i = 0; i < vertices.size(); i++) {
					fprintf(f, "<circle cx=\"%u\" cy=\"%u\" r=\"2\" stroke=\"black\" stroke-width=\"1\" fill = \"red\" />", (int)(vertices[i][0] * 4000), (int)(4000 - vertices[i][1] * 4000));
				}
				fprintf(f, "</g>\n");
			}


			fprintf(f, "</svg>\n");

			fclose(f);
		}

		mutable Vector tmpMemVector[64][2048];
		mutable int tmpMemInt[64][2048];

		// clip a cell by a bisector between site si and sj (possibly weighted)
		void clipCellByBissectorPlane(std::vector<Polygon>& cells, int si, int sj) const {
			#ifdef WITH_OMP
				int tid =  omp_get_thread_num();
			#else
				int tid = 0;
			#endif
			size_t Nv = cells[si].vertices.size();
			if (Nv == 0) return;

			const Vector diff = vertices[si] - vertices[sj];
			Vector M;
			//https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm
			if (weights.size() != 0) {
				double	weightsi = weights[si];
				double	weightsj = weights[sj];
				double off = (weightsi - weightsj) / diff.getNorm2();
				M = (vertices[si] + vertices[sj] - off * diff) * 0.5;
			} else {
				M = (vertices[si] + vertices[sj]) * 0.5;
			}

			int numNewVertices = 0;
			Vector* newVertices = &tmpMemVector[tid][0];
			int* newNeighbors = &tmpMemInt[tid][0];
			const Vector* pv2 = &cells[si].vertices[0];
			const Vector* pv1 = &cells[si].vertices[Nv - 1];

			Vector* pnewVertices = newVertices;
			int* pnewNeighbors = newNeighbors;

			for (size_t j = 0; j < Nv; j++, ++pv2) {

				const Vector& v1 = *pv1;
				const Vector& v2 = *pv2;

				double d1 = dot(M - v1, diff);

				if (dot(v2 - M, diff) > -1E-12) { // v2 inside			
					if (d1 > 1E-12) { // v1 outside
						const Vector v1v2 = v2 - v1;
						double t = d1 / dot(v1v2, diff);
						(*pnewVertices)[0] = v1[0] + t * v1v2[0];
						(*pnewVertices)[1] = v1[1] + t * v1v2[1];
						(*pnewNeighbors) = sj;
						numNewVertices++;
						++pnewVertices;
						++pnewNeighbors;
					}
					(*pnewVertices) = v2;
					(*pnewNeighbors) = cells[si].neighbors[j];
					numNewVertices++;
					++pnewVertices;
					++pnewNeighbors;
				} else {

					if (d1 < -1E-12) {
						const Vector v1v2 = v2 - v1;
						double d2 = dot(v1v2, diff);
						double t = d1 / d2;
						(*pnewVertices)[0] = v1[0] + t * v1v2[0];
						(*pnewVertices)[1] = v1[1] + t * v1v2[1];
						(*pnewNeighbors) = cells[si].neighbors[j];
						numNewVertices++;
						++pnewVertices;
						++pnewNeighbors;

					}
				}
				pv1 = pv2;
			}

			if (numNewVertices == 0) {
				cells[si].vertices.clear();
				cells[si].neighbors.clear();
				return;
			}

			cells[si].vertices.resize(numNewVertices);
			cells[si].neighbors.resize(numNewVertices);
			memcpy(&cells[si].vertices[0], newVertices, numNewVertices * sizeof(Vector));
			memcpy(&cells[si].neighbors[0], newNeighbors, numNewVertices * sizeof(int));
		}


		std::vector< std::vector<int> > neighbors; // for dual computation

		// from a given delaunay triangulation, find the corresponding voronoi (restricted to [0,1]^2
		void compute_dual() {

			Polygon square;
			square.vertices.resize(4);
			square.neighbors.resize(4, -1);
			square.vertices[0] = Vector(0., 0.);
			square.vertices[3] = Vector(1., 0.);
			square.vertices[2] = Vector(1., 1.);
			square.vertices[1] = Vector(0., 1.);

			// compute the one-ring
			neighbors.clear();
			std::vector<int> neighbor_template; neighbor_template.reserve(30);
			neighbors.resize(vertices.size(), neighbor_template);  // not sure if this works : initialize with a zero-sized vector but with reserved memory
			for (int i = 0; i < triangles.size(); i++) {
				neighbors[triangles[i].id[0]].push_back(triangles[i].id[1]);
				neighbors[triangles[i].id[0]].push_back(triangles[i].id[2]);

				neighbors[triangles[i].id[1]].push_back(triangles[i].id[0]);
				neighbors[triangles[i].id[1]].push_back(triangles[i].id[2]);

				neighbors[triangles[i].id[2]].push_back(triangles[i].id[0]);
				neighbors[triangles[i].id[2]].push_back(triangles[i].id[1]);
			}

			// we have inserted everything 3 times, so make it unique (faster than inserting to an std::set or unordered_set)
#ifdef WITH_OMP
#pragma omp parallel for
#endif
			for (int i = 0; i < vertices.size(); i++) {
				std::sort(neighbors[i].begin(), neighbors[i].end());
				auto last = std::unique(neighbors[i].begin(), neighbors[i].end());
				neighbors[i].erase(last, neighbors[i].end());
			}

			// for each Voronoi cell (to be computed), start with a square, and clip it by the bisectors returned by the Delaunay 
			voronoi.resize(vertices.size());
#ifdef WITH_OMP
#pragma omp parallel for 
#endif
			for (int i = 0; i < vertices.size(); i++) {
				
				if (neighbors[i].size() == 0) {
					voronoi[i].vertices.clear();
					continue;
				}
				voronoi[i] = square;

				for (auto it = neighbors[i].begin(); it != neighbors[i].end(); ++it) {
					clipCellByBissectorPlane(voronoi, i, *it);
				}

			}

		}

		// relative area error in PERCENT of the cell that has the most different from an area 1/N
		double worst_cell_area_relative_diff_percent() {
			if (voronoi.size() == 0) {
				std::cout << "worst_cell_area(): Voronoi not computed => compute it first" << std::endl;
				return -1;
			}

			double idealArea = 1. / vertices.size();
			double worstAreaDiff = 0;

			for (int i = 0; i < vertices.size(); i++) {
				double area = voronoi[i].area();
				double diff = fabs(idealArea - area);
				if (diff > worstAreaDiff) {
					worstAreaDiff = diff;
				}
			}
			return worstAreaDiff / idealArea * 100.;
		}

		// average relative area error in PERCENT 
		double avg_cell_area_relative_diff_percent() {
			if (voronoi.size() == 0) {
				std::cout << "avg_cell_area_relative_diff_percent(): Voronoi not computed => compute it first" << std::endl;
				return -1;
			}

			double idealArea = 1. / vertices.size();
			double sum = 0;

			for (int i = 0; i < vertices.size(); i++) {
				double area = voronoi[i].area();
				double diff = fabs(idealArea - area);
				sum += diff;
			}
			return sum/(vertices.size() * idealArea) * 100.;
		}

		std::vector<Polygon> voronoi;
		std::vector<Vector> vertices;
		std::vector<double> weights;
		std::vector<Triangle> triangles;

		std::vector<Vector> sorted_vertices;

	private:

		std::vector<double> sorted_weights;
		std::vector<double> sorted_heights; //unsorted heights do not exist

		std::vector<int> lastValid;


		TriangleP* lastvP;

		bool lifted;
	};




	class OptimalTransport2D {
	public:
		OptimalTransport2D(): has_density(false) {			
		};

		// densityW is the width (== height) of the image used as a density, stored in the density variable in row major
		OptimalTransport2D(const std::vector<double>& density, int densityW):density(density), densityW(densityW), has_density(true) {
		};

		void setDensity(const std::vector<double>& d, int dW)
		{
			density = d;
			densityW = dW;
			has_density = true;
		}


		double pointToIndex(double point, double size) {
			return std::floor(point * size);
		}
		
		double integrateDensityOverEdge(Vector p0, Vector p1) {

			if ((p0 - p1).getNorm2() == 0) return 0;

			p0[0] = std::max(1E-10, std::min(p0[0], 1. - 1E-10));
			p0[1] = std::max(1E-10, std::min(p0[1], 1. - 1E-10));
			p1[0] = std::max(1E-10, std::min(p1[0], 1. - 1E-10));
			p1[1] = std::max(1E-10, std::min(p1[1], 1. - 1E-10));

			Vector scaledP0 = Vector(p0[0] * densityW, p0[1] * densityW);
			Vector scaledP1 = Vector(p1[0] * densityW, p1[1] * densityW);

			double dx = scaledP1[0] - scaledP0[0];
			double dy = scaledP1[1] - scaledP0[1];
			double segNorm = std::sqrt(dx * dx + dy * dy);
			double stepX = dx > 0 ? 1 : -1;
			double stepY = dy > 0 ? 1 : -1;

			// tDeltaX, tDeltaY for crossing one pixel
			double tDeltaX = fabs(1.0 / dx);
			double tDeltaY = fabs(1.0 / dy);

			// Initial tMaxX, tMaxY for the first pixel's boundaries
			double tMaxX = (stepX > 0 ? ceil(scaledP0[0]) - scaledP0[0] : scaledP0[0] - floor(scaledP0[0])) * tDeltaX;
			double tMaxY = (stepY > 0 ? ceil(scaledP0[1]) - scaledP0[1] : scaledP0[1] - floor(scaledP0[1])) * tDeltaY;
			


			double sum = 0.0;
			int pixelX = floor(scaledP0[0]);
			int pixelY = floor(scaledP0[1]);
			double prevt = 0;
			double sumseg = 0;
			//double segltot = (scaledP1 - scaledP0).getNorm();
			while (true) {
				// Accumulate weighted pixel value
				if (pixelX >= 0 && pixelX < densityW && pixelY >= 0 && pixelY < densityW) {
					double segmentLength = segNorm * (std::min(tMaxX, tMaxY) - prevt);
					sumseg += segmentLength;
					sum += density[pixelY * densityW + pixelX] * segmentLength;
				}
				prevt = std::min(tMaxX, tMaxY);

				if (tMaxX < tMaxY) {
					tMaxX += tDeltaX;
					pixelX += stepX;
				} else {
					tMaxY += tDeltaY;
					pixelY += stepY;
				}

				// Check if we've reached the end of the line segment
				/*if ((dx > 0 && pixelX >= scaledP1[0]) || (dx < 0 && pixelX <= scaledP1[0]) ||
					(dy > 0 && pixelY >= scaledP1[1]) || (dy < 0 && pixelY <= scaledP1[1])) {
					break;
				}*/
				if ((dx > 0 && pixelX >= std::floor(scaledP1[0])) || (dx < 0 && pixelX <= std::ceil(scaledP1[0])) ||
					(dy > 0 && pixelY >= std::floor(scaledP1[1])) || (dy < 0 && pixelY <= std::ceil(scaledP1[1]))) {
					break;
				}
			}

			if (pixelX >= 0 && pixelX < densityW && pixelY >= 0 && pixelY < densityW) {
				double segmentLength = segNorm * (1.0 - prevt);
				sumseg += segmentLength;
				sum += density[pixelY * densityW + pixelX] * segmentLength;
			}

			return sum / densityW; // Correct the scale of the sum since we did not work on the [0,1]^2 square
		}


		// finally, more efficient to pre-store the Hessian than computing on the fly
		void precompute_hessian() {

#ifdef WITH_OMP
#pragma omp parallel for
#endif 
				for (int i = 0; i < N; i++) {

					double result = 0;
					double sum_contribs = 0;
					size_t num_nn = V.voronoi[i].neighbors.size();
					const int* nn = &V.voronoi[i].neighbors[0];
					const Vector* vtx = &V.voronoi[i].vertices[0];
					const Vector* prev_vtx = &V.voronoi[i].vertices[num_nn - 1];
					Vector curSample = V.vertices[i];

					HessianValues[i].clear();
					HessianValues[i].reserve(V.voronoi[i].vertices.size());

					double sumcontrib = 0;
					for (int j = 0; j < num_nn; j++, ++nn, ++vtx) {
						double edgeLength;
						//double etest;
						if (has_density) {
							edgeLength = integrateDensityOverEdge(*vtx, *prev_vtx);
							//etest = (*vtx - *prev_vtx).getNorm();
						} else {
							edgeLength = (*vtx - *prev_vtx).getNorm();
						}
						prev_vtx = vtx;
						if (*nn < 0) continue;


						double contrib = -edgeLength / (2. * (curSample - V.vertices[*nn]).getNorm());
						sumcontrib += contrib;

						//if (i < *nn) continue; // ONLY store lower triangular part !  => for some reason, this increases the final error, and complexifies parallelism
						HessianValues[i].push_back(std::pair<int, double>(*nn, contrib));
					}
					diagHessian[i] = -sumcontrib;

				
			}
		}

		std::vector<std::vector<std::pair<int, double> > > HessianValues;
		std::vector<double> diagHessian; // also used for preconditionning
		std::vector<double> r, p, z, Ap;

		// compressed row storage provides ~10% speeedup in mat-vec product
		std::vector<double> HessianValuesCRS_val;
		std::vector<unsigned int> HessianValuesCRS_col;
		std::vector<unsigned int> HessianValuesCRS_row;

		void hessian_to_CRS() {

			size_t nnz = 0;
			for (size_t i = 0; i < N; i++)
				nnz += HessianValues[i].size();

			HessianValuesCRS_val.resize(nnz);
			HessianValuesCRS_col.resize(nnz);
			HessianValuesCRS_row.resize(N + 1);

			size_t k = 0;
			HessianValuesCRS_row[0] = 0;
			size_t totalnnz = 0;
			for (int i = 0; i < N; i++) {
				totalnnz += HessianValues[i].size();
				for (int j = 0; j < HessianValues[i].size(); j++) {
					HessianValuesCRS_val[k] = HessianValues[i][j].second;
					HessianValuesCRS_col[k] = HessianValues[i][j].first;
					k++;
				}
				HessianValuesCRS_row[i + 1] = k;
			}

			// better load balancing of nnz between threads
			int cumNnz = 0;
			rowIdx[0] = 0;
			int block = 1;
			int maxThread = nb_threads;
			for (int i = 0; i < N; i++) {
				cumNnz+= HessianValues[i].size();
				if (cumNnz>=(block* totalnnz)/ maxThread) { // 100 elements over 3 threads would give 0-32 33-65 66-100  (33+33+34)
					rowIdx[block] = i;
					block++;
				}				
			}
			rowIdx[maxThread] = N;

		}

		// for debugging purposes 
		void saveHessianSparse(std::string filename) {

			FILE* f = fopen(filename.c_str(), "w+");
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < HessianValues[i].size(); j++) {
					fprintf(f, "%u %u %e\n", i, HessianValues[i][j].first, HessianValues[i][j].second);
				}
			}
			fclose(f);
		}

		size_t rowIdx[128];


		void hessian_mult(double* rhs, double* result) {

			//memset(result, 0, N * sizeof(double));
#ifdef WITH_OMP
#pragma omp parallel
#endif 
			{
				#ifdef WITH_OMP
					int tid = omp_get_thread_num();
				#else
					int tid = 0;
				#endif

				if (tid <= nb_threads) {
					int beginIdx = rowIdx[tid];
					int endIdx = rowIdx[tid + 1];
					for (int i = beginIdx; i < endIdx; i++) {

						double sum = 0;
						unsigned int startingVal = HessianValuesCRS_row[i];
						int num_nn = HessianValuesCRS_row[i + 1] - startingVal;
						const double* H = &HessianValuesCRS_val[startingVal];
						const unsigned int* cols = &HessianValuesCRS_col[startingVal];
						//double rhs_i = rhs[i];
						int nn2 = num_nn / 2;
						int nn4 = num_nn / 4;

						for (int j = 0; j < nn4; j++, H += 4, cols += 4) {
							sum += *H * rhs[*cols];
							sum += *(H + 1) * rhs[*(cols + 1)];
							sum += *(H + 2) * rhs[*(cols + 2)];
							sum += *(H + 3) * rhs[*(cols + 3)];
							//result[*cols] += *H * rhs_i;  // if only triangular part stored ; need to remove parallel for then, or have local thread storage for result
						}
						if (nn2 * 2 != num_nn) { // n%4 == 1 or 3
							if (nn4 * 4 == num_nn - 1) { // n%4 == 1
								sum += *H * rhs[*cols];
							} else {
								sum += *H * rhs[*cols]; ++H; ++cols;
								sum += *H * rhs[*cols]; ++H; ++cols;
								sum += *H * rhs[*cols];
							}
						} else {
							if (nn4 * 4 != num_nn) {  // n%4 == 2
								sum += *H * rhs[*cols]; ++H; ++cols;
								sum += *H * rhs[*cols];
							} // else // n%4 = 0
						}

						result[i] = sum + diagHessian[i] * rhs[i];

					}
				}
			}
		}




		void conjugate_gradient_solve(double* rhs, double* result) { // result initialized ( = x0 )
			r.resize(N);
			p.resize(N);
			z.resize(N);
			Ap.resize(N);
			HessianValues.resize(N);
			diagHessian.resize(N);

			precompute_hessian();
			hessian_to_CRS();
			
			hessian_mult(result, &Ap[0]);

			for (int i = 0; i < N; i++) {
				r[i] = rhs[i] - Ap[i];
				z[i] = r[i] / diagHessian[i]; // Jacobi preconditionning
			}
			p = z;


			for (int k = 0; k < std::min(N, N / 4 + 2); k++) { //had to limit the number of iterations for edgy cases where it starts to blow up

				hessian_mult(&p[0], &Ap[0]);
				double rz = 0, pAp = 0;

#ifdef WITH_OMP
#pragma omp parallel for reduction(+ : rz, pAp) 
#endif
				for (int i = 0; i < N; i++) {
					rz += r[i] * z[i];
					pAp += p[i] * Ap[i];
				}
				double alpha = rz / pAp;
				double rzb = 0;
				double rr = 0;

				/*double* presult = result;
				double* pr = &r[0];
				double* pp = &p[0];
				double* pz = &z[0];
				double* pointAp = &Ap[0];
				for (int i = 0; i < N; i++) {
					*presult += alpha * *pp;
					*pr -= alpha * *pointAp;
					*pz = *pr / diagHessian[i];
					rzb += *pr * *pz;
					rr += sqr(*pr);

					presult++;
					pr++;
					pp++;
					pz++;
					pointAp++;
				}*/
#ifdef WITH_OMP
#pragma omp parallel for reduction(+ : rzb, rr)
#endif
				for (int i = 0; i < N; i++) {
					result[i] += alpha * p[i];
					r[i] -= alpha * Ap[i];
					z[i] = r[i] / diagHessian[i];

					rzb += r[i] * z[i];
					rr += sqr(r[i]);
				}
				if (!has_density && rr < 1E-12) break; // FOR UNIFORM DENSITIES !
				if (has_density && rr < 1E-3) break; // FOR NON UNIFORM DENSITIES !

				double beta = rzb / rz;

				#ifdef WITH_OMP
				#pragma omp parallel for 
				#endif
				for (int i = 0; i < N; i++) {
					p[i] = z[i] + beta * p[i];
				}
			}

		}

		// returns the worst relative difference in area, in percentage (1 = 1% area error w.r.t a uniform 1./N). Also used as stopping criterion
		double newton_optimize(int max_Newton_iter, double worst_area_relative_threshold_percent) {

			N = V.vertices.size();
			V.weights.resize(N);
			std::fill(V.weights.begin(), V.weights.end(), 0.0);
			
			double lambda = 1;
			double worstarea = 0;
			std::vector<double> grad(N), invGrad(N, 0.), allAreas(N, 0.);
			for (int iter = 0; iter < max_Newton_iter; iter++) {
			start:
				//V.clear();
				//V.build();
				V.compute_delaunay(iter == 0);
				V.compute_dual();
				
				worstarea = 0;
#ifdef WITH_OMP
#pragma omp parallel for
#endif
				for (int i = 0; i < N; i++) {
					double a;
					Vector barycenter;
					//double atest;
					if (has_density) {
						a = V.voronoi[i].weighted_area(&density[0], densityW, barycenter);
						//atest = V.voronoi[i].area();

					} else {
						a = V.voronoi[i].area();
					}
					allAreas[i] = a;
				}

				for (int i = 0; i < N; i++) {
					
					double a = allAreas[i];


					if (a <= 0) { // if an empty cell is detected, we roll back half the step size and reduce the step size by half. See Gallouet&Merigot
						//std::cout << "new lambda:"<<lambda*0.5 << std::endl;
						lambda *= 0.5;
						if (lambda < 1E-9) {
							std::cout << "ERROR during optimization ; resulting value is unreliable !" << std::endl;
							return -1;
						}
						for (int j = 0; j < N; j++) {
							V.weights[j] -= lambda * invGrad[j];
						}

						goto start; // boo me for that goto as much as you want, I'm a punk
						break;
					}
					grad[i] = 1.0 / N - a;
					if (fabs(grad[i]) > worstarea) worstarea = fabs(grad[i]);
				}
				//std::cout << worstarea * N * 100 << std::endl;
				if (worstarea * N * 100 < worst_area_relative_threshold_percent) {
					//std::cout << "optimization successfully stopped after " << iter << " Newton iterations" << std::endl;
					break;
				}
				
				//memset(&invGrad[0], 0, invGrad.size() * sizeof(double));
				conjugate_gradient_solve(&grad[0], &invGrad[0]);

				for (int i = 0; i < N; i++) {
					V.weights[i] += lambda * invGrad[i];
				}
			}

			// we may have reached here after a Newton step ; we recompute the power diagram
			V.compute_delaunay(false);
			V.compute_dual();


			//std::cout << worstarea * N * 100 << "%" << std::endl;
			return worstarea * N * 100;
		}


		double optimize(int max_Newton_iter = 100, double worst_area_relative_threshold_percent = 0.5, double* resulting_worst_area_error = NULL) {

			// ugly hack ; while the Delaunay is most efficiently computed when vertices are spatially sorted (hence the Bowyer::sort_vertices method), 
			// this is *also* the case for the Newton solve, since this reduces the matrix profile and improves cache coherence. So I'm ultimately calling the Bowyer::sort here.
			N = V.vertices.size();
#ifdef WITH_OMP
	#pragma omp parallel
#endif			
			{
				#ifdef WITH_OMP
					#pragma omp critical
				#endif
				{
				#ifdef WITH_OMP
					nb_threads = omp_get_num_threads();
				#else
					nb_threads = 1;
				#endif
				}
			}
			#ifdef WITH_OMP
				if (N < nb_threads) nb_threads = N;
			#endif


			std::vector<Vector> saved_vertices = V.vertices;			
			V.sort_vertices();
			std::vector<int> perm_vector = V.perm;
			V.vertices = V.sorted_vertices;

			V.weights.resize(N);
			std::fill(V.weights.begin(), V.weights.end(), 0.0); // need to start with voronoi: it guarantees no cells are empty

			double worst_error = newton_optimize(max_Newton_iter, worst_area_relative_threshold_percent);

			if (resulting_worst_area_error)
				*resulting_worst_area_error = worst_error;

			// I'm now restoring back the vertices and puting weights in order
			V.vertices = saved_vertices;
			std::vector<double> reordered_weights(V.vertices.size());
			for (int i = 0; i < V.vertices.size(); i++) {
				reordered_weights[perm_vector[i]] = V.weights[i];
			}
			V.weights = reordered_weights;

			V.compute_delaunay(false);
			V.compute_dual();
			double sI = 0;
			double sItestsc = 0;
			if (has_density) {
				Vector barycenter;
#ifdef WITH_OMP
#pragma omp parallel for reduction(+ : sI)
#endif
				for (int i = 0; i < N; i++) {
					sI += V.voronoi[i].weighted_area(&density[0], densityW, barycenter, true, V.vertices[i]);
					//sItest += V.voronoi[i].integrateSquaredDistance2D(V.vertices[i]);
				}				
			} else {
				for (size_t i = 0; i < N; i++) {
					sI += V.voronoi[i].integrateSquaredDistance2D(V.vertices[i]);
				}
			}
			
			return sI;
		}

		void ot_lloyd(int n_lloyd_iter = 100) {

			for (int iter = 0; iter < n_lloyd_iter; iter++) {
				std::cout << "performing Lloyd iteration " << iter <<" over "<< n_lloyd_iter << std::endl;
				optimize(100);

				if (has_density) {
					for (int i = 0; i < V.vertices.size(); i++) {
						Vector barycenter;
						double a = V.voronoi[i].weighted_area(&density[0], densityW, barycenter);
						V.vertices[i] = barycenter;
					}
				} else {
					for (int i = 0; i < V.vertices.size(); i++) {
						V.vertices[i] = V.voronoi[i].centroid();
					}
				}

			}

		}




		Bowyer2D V;
		size_t N;
		int nb_threads;
		bool has_density;
		std::vector<double> density;
		int densityW;
	};

};