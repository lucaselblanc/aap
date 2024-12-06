// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC/Kang-2

#include "defs.h"
#include <vector>
#include "Ec.h"
#include "utils.h"
#include <thread>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include <chrono>
#include <iostream>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//GENERAL NOTE:
//This is Part 2 of my research "Solving ECDLP with Kangaroos".
//so now we have SOTA method with K=1.15 but there is an issue with looping:
//when 2^DP_BITS is much more than JMP_CNT, simple loop handling restarts kangaroos before they reach a DP, i.e. breaks a chain of kangaroo jumps.
//Database(DB) of DPs grows as normal, but every DP represents much less points than it must (DP_BITS).
//As a result, K is much worse though DB looks good.
//Proposed here advanced loop handling does not break a chain of jumps and reduces the number of large loops for high ranges so K is the same as it must be.
//the difference between Simple and Advanced methods grows with growing RANGE_BITS and DP_BITS

#define RANGE_BITS			(40 + 0)
#define DP_BITS				(8)

//Usually, the higher the range, the more number of kangaroos is used (because more devices are used for solving), so their paths grow a bit slower than sqrt(range) and it helps to keep K stable.
//but you can use fixed number of kangaroos to confirm that everything works properly for any range even without increasing the number of kangaroos
//#define KANG_CNT			(1 << (int)(0.2 * RANGE_BITS))
#define KANG_CNT			512

#define JMP_CNT				(1024)

#define CPU_THR_CNT			(63)
#define POINTS_CNT			(500)
#define OLD_LEN				(16)


////advanced-only settings
//it's a good idea to have a large L2 table, but not mandatory
#define JMP_CNT2			(1*1024)

//we can escape from large loops, but there is no large loops detection on GPU fast implementation, so behaviour will be different
//but if we dont escape large loops, stats will be incorrect, same loops will be counted many times
//so uncomment it for stats, comment it for same behaviour as on GPU
// #define ESCAPE_FROM_LARGE_LOOPS

//loop stats: len of every list
#define MD_LEN		16
//loop stats: number of lists
#define MD_CNT		10
// MD_LEN 16 and MD_CNT 10 mean we can handle loops up to 2^40

//if defined, we stop searching a key after specified number of iterations so we can check loop stats for high RANGE_BITS
// #define SYNTHETIC_TEST
#ifdef SYNTHETIC_TEST
	#define MAX_TOTAL_ITERS		(100 * 1000 * 1000ull)
	#define RANGE_BITS			60
	#define POINTS_CNT			CPU_THR_CNT
	#define KANG_CNT			1
	#define ESCAPE_FROM_LARGE_LOOPS
#endif

// Define para substituição de BYTE //////////////REMOVE
using BYTE = unsigned char;

using HANDLE = std::thread::native_handle_type;//////////////REMOVE

EcInt BigValue;

uint64_t GetTickCount64() {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    return static_cast<uint64_t>(duration.count());
}

void Sleep(unsigned int milliseconds) {
    usleep(milliseconds * 1000);
}

struct EcJMP
{
	EcPoint p;
	EcInt dist;
};
EcJMP EcJumps[JMP_CNT];
EcJMP EcJumps2[JMP_CNT2];

struct EcKang
{
	EcPoint p;
	EcInt dist;
	int iter; //iters without new DP
	u64 total_iters;
};
typedef std::vector <EcKang> EcKangs;

EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt Int_HalfRange;
Ec ec;

volatile long ThrCnt;
volatile long SolvedCnt;
volatile long ToSolveCnt;

struct TThrRec
{
	HANDLE hThread;
	CExpKangDlg* obj;
	u64 iters;
	int thr_ind;
};

#define TAME	0
#define WILD	1
#define WILD2	2

struct TDB_Rec
{
	BYTE x[12];
	BYTE d[12];
	int type; //0 - tame, 1 - wild1, 2 - wild2
};

bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg)
{
	if (IsNeg)
		t.Neg();
	if (TameType == TAME)
	{
		EcInt pk = t;
		pk.Sub(w);
		EcInt sv = pk;
		pk.Add(Int_HalfRange);
		EcPoint P = ec.MultiplyG(pk);
		if (P.IsEqual(pnt))
			return true;
		pk = sv;
		pk.Neg();
		pk.Add(Int_HalfRange);
		P = ec.MultiplyG(pk);
		return P.IsEqual(pnt);
	}
	else
	{
		EcInt pk = t;
		pk.Sub(w);
		if (pk.data[4] >> 63)
			pk.Neg();
		pk.ShiftRight(1);
		EcInt sv = pk;
		pk.Add(Int_HalfRange);
		EcPoint P = ec.MultiplyG(pk);
		if (P.IsEqual(pnt))
			return true;
		pk = sv;
		pk.Neg();
		pk.Add(Int_HalfRange);
		P = ec.MultiplyG(pk);
		return P.IsEqual(pnt);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

u32 thr_proc_sota_simple(void* data)
{
	TThrRec* rec = (TThrRec*)data;
	rec->iters = 0;
	EcKangs kangs;
	kangs.resize(KANG_CNT);
	u32 DPmask = (1 << DP_BITS) - 1;
	TFastBase* db = new TFastBase();
	db->Init(sizeof(TDB_Rec::x), sizeof(TDB_Rec), 0, 0);

	u64* old = (u64*)malloc(OLD_LEN * 8 * KANG_CNT);
	int max_iters = (1 << DP_BITS) * 20;
	while (1)
	{
		if (InterlockedDecrement(&ToSolveCnt) < 0)
			break;
		EcInt KToSolve;
		EcPoint PointToSolve;
		EcPoint NegPointToSolve;

		memset(old, 0, OLD_LEN * 8 * KANG_CNT);
		KToSolve.RndBits(RANGE_BITS);

		for (int i = 0; i < KANG_CNT; i++)
		{
			if (i < KANG_CNT / 3)
				kangs[i].dist.RndBits(RANGE_BITS - 4);
			else
			{
				kangs[i].dist.RndBits(RANGE_BITS - 1);
				kangs[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
			}
		}

		PointToSolve = ec.MultiplyG(KToSolve);
		EcPoint Pnt1 = ec.AddPoints(PointToSolve, Pnt_NegHalfRange);
		EcPoint Pnt2 = Pnt1;
		Pnt2.y.NegModP();
		for (int i = 0; i < KANG_CNT; i++)
		{
			kangs[i].p = ec.MultiplyG(kangs[i].dist);
			kangs[i].iter = 0;
		}

		for (int i = KANG_CNT / 3; i < 2 * KANG_CNT / 3; i++)
			kangs[i].p = ec.AddPoints(kangs[i].p, Pnt1);
		for (int i = 2 * KANG_CNT / 3; i < KANG_CNT; i++)
			kangs[i].p = ec.AddPoints(kangs[i].p, Pnt2);

		bool found = false;
		while (!found)
		{
			for (int i = 0; i < KANG_CNT; i++)
			{
				bool inv = (kangs[i].p.y.data[0] & 1);
				bool cycled = false;
				for (int j = 0; j < OLD_LEN; j++)
					if (old[OLD_LEN * i + j] == kangs[i].dist.data[0])
					{
						cycled = true;
						break;
					}
				old[OLD_LEN * i + (kangs[i].iter % OLD_LEN)] = kangs[i].dist.data[0];
				kangs[i].iter++;
				if (kangs[i].iter > max_iters)
					cycled = true;
				if (cycled)
				{
					if (i < KANG_CNT / 3)
						kangs[i].dist.RndBits(RANGE_BITS - 4);
					else
					{
						kangs[i].dist.RndBits(RANGE_BITS - 1);
						kangs[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
					}

					kangs[i].iter = 0;
					kangs[i].p = ec.MultiplyG(kangs[i].dist);
					if (i >= KANG_CNT / 3)
					{
						if (i < 2 * KANG_CNT / 3)
							kangs[i].p = ec.AddPoints(kangs[i].p, Pnt1);
						else
							kangs[i].p = ec.AddPoints(kangs[i].p, Pnt2);
					}
					memset(&old[OLD_LEN * i], 0, 8 * OLD_LEN);
					continue;
				}

				int jmp_ind = kangs[i].p.x.data[0] % JMP_CNT;
				EcPoint AddP = EcJumps[jmp_ind].p;
				if (!inv)
				{
					kangs[i].p = ec.AddPoints(kangs[i].p, AddP);
					kangs[i].dist.Add(EcJumps[jmp_ind].dist);
				}
				else
				{
					AddP.y.NegModP();
					kangs[i].p = ec.AddPoints(kangs[i].p, AddP);
					kangs[i].dist.Sub(EcJumps[jmp_ind].dist);
				}
				rec->iters++;

				if (kangs[i].p.x.data[0] & DPmask)
					continue;

				TDB_Rec nrec;
				memcpy(nrec.x, kangs[i].p.x.data, 12);
				memcpy(nrec.d, kangs[i].dist.data, 12);
				if (i < KANG_CNT / 3)
					nrec.type = TAME;
				else
					if (i < 2 * KANG_CNT / 3)
						nrec.type = WILD;
					else
						nrec.type = WILD2;

				TDB_Rec* pref = (TDB_Rec*)db->FindOrAddDataBlock((uint8_t*)&nrec, sizeof(nrec));
				if (pref)
				{
					if (pref->type == nrec.type)
					{
						if (pref->type == TAME)
							continue;

						//if it's wild, we can find the key from the same type if distances are different
						if (*(u64*)pref->d == *(u64*)nrec.d)
							continue;
						//else
						//std::cout << "key found by same wild!" << std::endl;
					}

					EcInt w, t;
					int TameType, WildType;
					if (pref->type != TAME)
					{
						memcpy(w.data, pref->d, sizeof(pref->d));
						if (pref->d[11] == 0xFF) memset(((uint8_t*)w.data) + 12, 0xFF, 28);
						memcpy(t.data, nrec.d, sizeof(nrec.d));
						if (nrec.d[11] == 0xFF) memset(((uint8_t*)t.data) + 12, 0xFF, 28);
						TameType = nrec.type;
						WildType = pref->type;
					}
					else
					{
						memcpy(w.data, nrec.d, sizeof(nrec.d));
						if (nrec.d[11] == 0xFF) memset(((uint8_t*)w.data) + 12, 0xFF, 28);
						memcpy(t.data, pref->d, sizeof(pref->d));
						if (pref->d[11] == 0xFF) memset(((uint8_t*)t.data) + 12, 0xFF, 28);
						TameType = TAME;
						WildType = nrec.type;
					}

					bool res = Collision_SOTA(PointToSolve, t, TameType, w, WildType, false) || Collision_SOTA(PointToSolve, t, TameType, w, WildType, true);
					if (!res)
					{
						//bool w12 = ((pref->type == WILD) && (nrec.type == WILD2)) || ((pref->type == WILD2) && (nrec.type == WILD));
						//if (w12) //in rare cases WILD and WILD2 can collide in mirror, in this case there is no way to find K
						//std::cout << "W1 and W2 collides in mirror!" << std::endl;
						continue;
					}
					found = true;
					break;
				}
				else
				{
					kangs[i].iter = 0;
					memset(&old[OLD_LEN * i], 0, 8 * OLD_LEN);
				}
			}
		}
		db->Clear(false);
		InterlockedIncrement(&SolvedCnt);
	}
	free(old);
	delete db;
	InterlockedDecrement(&ThrCnt);
	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

i64 loop_stats[MD_CNT + 1][MD_LEN];
i64 large_loop_cnt; //number of large loops (>=MD_LEN)
u32 L2loop_cnt; //number of L2 size-2 loops

typedef std::vector <EcInt> EcInts;

//add a minimal distance value to a list and check if it's already there
int add_to_md(EcInt* md, int ind, EcInt* val)
{
	if (!val->IsEqual(BigValue))
		for (int i = ind - 1 + MD_LEN; i >= ind; i--)
			if (md[i % MD_LEN].IsEqual(*val))
			{
				md[ind] = *val;
				return i % MD_LEN;
			}
	md[ind] = *val;
	return -1;
}

//find minimal value in the list
EcInt get_min_md(EcInt* md)
{
	EcInt res = md[0];
	for (int i = 1; i < MD_LEN; i++)
		if (md[i].IsLessThanI(res))
			res = md[i];
	return res;
}

//clear all values of a loop starting from "level" list and higher
void clear_loop(EcInt* bg, int level, int start, int end)
{
	EcInt* md = bg + level * MD_LEN;
	if (start >= end)
		end += MD_LEN;
	for (int i = start; i <= end; i++)
	{
		for (int j = level + 1; j < MD_CNT; j++)
		{
			EcInt* md2 = bg + j * MD_LEN;
			for (int k = 0; k < MD_LEN; k++)
				if (md2[k].IsEqual(md[i % MD_LEN]))
					md2[k] = BigValue;
		}
		md[i % MD_LEN] = BigValue;
	}
}

u32 thr_proc_sota_advanced(void* data)
{
	TThrRec* rec = (TThrRec*)data;
	rec->iters = 0;
	EcKangs kangs;
	kangs.resize(KANG_CNT);
	u32 DPmask = (1 << DP_BITS) - 1;
	TFastBase* db = new TFastBase();
	db->Init(sizeof(TDB_Rec::x), sizeof(TDB_Rec), 0, 0);

	EcInts** mds = (EcInts**)malloc(KANG_CNT * 8);
	for (int i = 0; i < KANG_CNT; i++)
	{
		mds[i] = new EcInts();
		mds[i]->resize(MD_LEN * MD_CNT);
	}
	u64* mdc = (u64*)malloc(KANG_CNT * 8);

	while (1)
	{
		if (InterlockedDecrement(&ToSolveCnt) < 0)
			break;
		EcInt KToSolve;
		EcPoint PointToSolve;
		EcPoint NegPointToSolve;

		KToSolve.RndBits(RANGE_BITS);

		for (int i = 0; i < KANG_CNT; i++)
		{
			if (i < KANG_CNT / 3)
				kangs[i].dist.RndBits(RANGE_BITS - 4);
			else
			{
				kangs[i].dist.RndBits(RANGE_BITS - 1);
				kangs[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
			}
		}

		PointToSolve = ec.MultiplyG(KToSolve);
		EcPoint Pnt1 = ec.AddPoints(PointToSolve, Pnt_NegHalfRange);
		EcPoint Pnt2 = Pnt1;
		Pnt2.y.NegModP();
		for (int i = 0; i < KANG_CNT; i++)
		{
			kangs[i].p = ec.MultiplyG(kangs[i].dist);
			kangs[i].total_iters = 0;
		}

		for (int i = KANG_CNT / 3; i < 2 * KANG_CNT / 3; i++)
			kangs[i].p = ec.AddPoints(kangs[i].p, Pnt1);
		for (int i = 2 * KANG_CNT / 3; i < KANG_CNT; i++)
			kangs[i].p = ec.AddPoints(kangs[i].p, Pnt2);

		EcPoint l2pnts[KANG_CNT];
		for (int i = 0; i < KANG_CNT; i++)
		{
			EcInts* md = mds[i];
			for (int j = 0; j < MD_LEN * MD_CNT; j++)
				md->at(j) = BigValue;
			mdc[i] = 0;
			l2pnts[i].x.Set(0);
			l2pnts[i].y.Set(0);
		}

		bool found = false;
		while (!found)
		{
			for (int i = 0; i < KANG_CNT; i++)
			{
				EcInts* md = mds[i];

				//we can imitate successful solving so we can check loop stats for high RANGE_BITS
#ifdef SYNTHETIC_TEST
					if (kangs[i].total_iters > MAX_TOTAL_ITERS)
					{
						std::cout << "Imitate successful solving!" << std::endl;
						found = true;
						break;
					}
#endif

				//check first list for small loops
				int loop_list_ind = -1;
				EcInt* bg = &md->at(0);
				int n = mdc[i] % MD_LEN;
				int ind = add_to_md(bg, n, &kangs[i].dist);
				if (ind >= 0)
				{
					clear_loop(bg, 0, ind, n);
					int dif;
					if (n > ind)
						dif = n - ind;
					else
						dif = MD_LEN - (ind - n);
					if (dif < MD_LEN)
						InterlockedIncrement64(&loop_stats[0][dif]);
					else
						InterlockedIncrement64(&loop_stats[1][1]);
					loop_list_ind = 0;
				}
				//check other lists for large loops
				u64 r = MD_LEN;
				for (int k = 0; k < MD_CNT - 1; k++)
				{
					if (((mdc[i] + 1) % r) == 0)
					{
						EcInt mn = get_min_md(bg + k * MD_LEN);
						int n = (mdc[i] / r) % MD_LEN;
						int ind = add_to_md(bg + (k + 1) * MD_LEN, n, &mn);
						if (ind >= 0)
						{
							InterlockedIncrement64(&large_loop_cnt);
							clear_loop(bg, k + 1, ind, n);
#ifdef ESCAPE_FROM_LARGE_LOOPS
							int dif;
							if (n > ind)
								dif = n - ind;
							else
								dif = MD_LEN - (ind - n);
							if (dif < MD_LEN)
								InterlockedIncrement64(&loop_stats[k + 1][dif]);
							else
								InterlockedIncrement64(&loop_stats[k + 2][1]);

							if (loop_list_ind < 0)
								loop_list_ind = k + 1;
#endif
						}
					}
					r *= MD_LEN;
				}			

				if (loop_list_ind >= 0) //exit from loop
				{			
					int l2detected = 0;
					EcPoint sv;
					while (1)
					{				
						sv = kangs[i].p;
						bool inv = (kangs[i].p.y.data[0] & 1);
						int jmp_ind = (kangs[i].p.x.data[0] + l2detected) % JMP_CNT2; //to break L2 loop we use next jump in the list
						EcPoint AddP = EcJumps2[jmp_ind].p;
						if (!inv)
						{
							kangs[i].p = ec.AddPoints(kangs[i].p, AddP);
							kangs[i].dist.Add(EcJumps2[jmp_ind].dist);
						}
						else
						{
							AddP.y.NegModP();
							kangs[i].p = ec.AddPoints(kangs[i].p, AddP);
							kangs[i].dist.Sub(EcJumps2[jmp_ind].dist);
						}
						if (l2pnts[i].IsEqual(kangs[i].p)) //L2 loop, rare
						{
							//std::cout << "L2 loop detected!" << std::endl;
							l2detected = 1; //to break L2-loop just use next jump, it will not break the chain
							InterlockedIncrement(&L2loop_cnt);

							//to confirm that L2 escaping works, disable these lines and set RANGE_BITS=44 JMP_CNT=JMP_CNT2=32 for a lot of L2 loops
							rec->iters++;
							kangs[i].total_iters++;
							continue; 
						}
						break;
					}
					l2pnts[i] = sv;
				}
				else
				{				
					bool inv = (kangs[i].p.y.data[0] & 1);
					int jmp_ind = kangs[i].p.x.data[0] % JMP_CNT;
					EcPoint AddP = EcJumps[jmp_ind].p;
					if (!inv)
					{
						kangs[i].p = ec.AddPoints(kangs[i].p, AddP);
						kangs[i].dist.Add(EcJumps[jmp_ind].dist);
					}
					else
					{
						AddP.y.NegModP();
						kangs[i].p = ec.AddPoints(kangs[i].p, AddP);
						kangs[i].dist.Sub(EcJumps[jmp_ind].dist);
					}
				}
				mdc[i]++;
				rec->iters++;
				kangs[i].total_iters++;

				if (kangs[i].p.x.data[0] & DPmask)
					continue;

#ifdef SYNTHETIC_TEST
				continue;
#endif

				TDB_Rec nrec;
				memcpy(nrec.x, kangs[i].p.x.data, 12);
				memcpy(nrec.d, kangs[i].dist.data, 12);
				if (i < KANG_CNT / 3)
					nrec.type = TAME;
				else
					if (i < 2 * KANG_CNT / 3)
						nrec.type = WILD;
					else
						nrec.type = WILD2;

				TDB_Rec* pref = (TDB_Rec*)db->FindOrAddDataBlock((uint8_t*)&nrec, sizeof(nrec));//BYTE
				if (pref)
				{
					if (pref->type == nrec.type)
					{
						if (pref->type == TAME)
							continue;

						//if it's wild, we can find the key from the same type if distances are different
						if (*(u64*)pref->d == *(u64*)nrec.d)
							continue;
						//else
						//std::cout << "key found by same wild!" << std::endl;
					}

					EcInt w, t;
					int TameType, WildType;
					if (pref->type != TAME)
					{
						memcpy(w.data, pref->d, sizeof(pref->d));
						if (pref->d[11] == 0xFF) memset(((uint8_t*)w.data) + 12, 0xFF, 28);
						memcpy(t.data, nrec.d, sizeof(nrec.d));
						if (nrec.d[11] == 0xFF) memset(((uint8_t*)t.data) + 12, 0xFF, 28);
						TameType = nrec.type;
						WildType = pref->type;
					}
					else
					{
						memcpy(w.data, nrec.d, sizeof(nrec.d));
						if (nrec.d[11] == 0xFF) memset(((uint8_t*)w.data) + 12, 0xFF, 28);
						memcpy(t.data, pref->d, sizeof(pref->d));
						if (pref->d[11] == 0xFF) memset(((uint8_t*)t.data) + 12, 0xFF, 28);
						TameType = TAME;
						WildType = nrec.type;
					}

					bool res = Collision_SOTA(PointToSolve, t, TameType, w, WildType, false) || Collision_SOTA(PointToSolve, t, TameType, w, WildType, true);
					if (!res)
					{
						bool w12 = ((pref->type == WILD) && (nrec.type == WILD2)) || ((pref->type == WILD2) && (nrec.type == WILD));
						if (w12) //in rare cases WILD and WILD2 can collide in mirror, in this case there is no way to find K
							;// std::cout << "W1 and W2 collides in mirror!" << std::endl;
						else
							std::cout << "Error!" << std::endl;
						continue;
					}
					found = true;
					break;
				}
			}
		}
		db->Clear(false);
		InterlockedIncrement(&SolvedCnt);
	}

	////
	for (int i = 0; i < KANG_CNT; i++)
		delete mds[i];
	free(mds);
	free(mdc);
	////

	delete db;
	InterlockedDecrement(&ThrCnt);
	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define METHOD_SIMPLE		0
#define METHOD_ADVANCED		1

char* names[] = {"SIMPLE", "ADVANCED"};

void Prepare(int Method)
{
	EcInt minjump, t;
	minjump.Set(1);
	minjump.ShiftLeft(RANGE_BITS / 2 + 3);	
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps[i].dist.Add(t);
		EcJumps[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps[i].p = ec.MultiplyG(EcJumps[i].dist);	
	}

	//prepare second jump table
	if (Method == METHOD_ADVANCED)
	{
		minjump.Set(1);
		minjump.ShiftLeft(RANGE_BITS - 10); //large jumps. Must be almost RANGE_BITS
		for (int i = 0; i < JMP_CNT2; i++)
		{
			EcJumps2[i].dist = minjump;
			t.RndMax(minjump);
			EcJumps2[i].dist.Add(t);
			EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
			EcJumps2[i].p = ec.MultiplyG(EcJumps2[i].dist);
		}
	}

	Int_HalfRange.Set(1);
	Int_HalfRange.ShiftLeft(RANGE_BITS - 1);
	Pnt_HalfRange = ec.MultiplyG(Int_HalfRange);
	Pnt_NegHalfRange = Pnt_HalfRange;
	Pnt_NegHalfRange.y.NegModP();

	memset((void*)loop_stats, 0, sizeof(loop_stats));
	large_loop_cnt = 0;
	L2loop_cnt = 0;
	BigValue.Set(1);
	BigValue.ShiftLeft(256);
}

void TestKangaroo(int Method)
{
	if (ThrCnt)
		return;
	if (RANGE_BITS < 40)
	{
		std::cout << "RANGE_BITS must be at least 40 bit, otherwise some formulas will stop work properly!" << std::endl;
		return;
	}

	std::cout << "Started, please wait..." << std::endl;
	SetRndSeed(0);
	Prepare(Method);
	//disable this line if you want exactly same results every time (may vary anyway for multithreading)
//	SetRndSeed(GetTickCount64());
	SolvedCnt = 0;
	TThrRec recs[CPU_THR_CNT];
	ThrCnt = CPU_THR_CNT;
	ToSolveCnt = POINTS_CNT;
	u64 tm = GetTickCount64();
	for (int i = 0; i < CPU_THR_CNT; i++)
	{
		u32 ThreadID;
		u32 (*thr_proc_ptr)(void*);
		switch (Method)
		{
		case METHOD_SIMPLE:
			thr_proc_ptr = thr_proc_sota_simple;
			break;
		case METHOD_ADVANCED:
			thr_proc_ptr = thr_proc_sota_advanced;
			break;
		default:
			return;
		}
		recs[i].hThread = (HANDLE)_beginthreadex(NULL, 0, thr_proc_ptr, (void*)&recs[i], 0, &ThreadID);
	}
	char s[300];
	while (ThrCnt)
	{
		sprintf(s, "Threads: %d. Solved: %d of %d", ThrCnt, SolvedCnt, POINTS_CNT);
		Sleep(100);
	}

	tm = GetTickCount64() - tm;
	
	sprintf(s, "Total time: %d sec", (int)(tm/1000));
	std::cout << s << std::endl;
	size_t iters_sum = recs[0].iters;
	for (int i = 1; i < CPU_THR_CNT; i++)
		iters_sum += recs[i].iters;

	size_t aver = iters_sum / POINTS_CNT;
	sprintf(s, "Total jumps for %d points: %llu", POINTS_CNT, iters_sum);
	std::cout << s << std::endl;
	sprintf(s, "Average jumps per point: %llu. Average jumps per kangaroo: %llu", aver, aver / KANG_CNT);
	std::cout << s << std::endl;
	double root = std::pow(2, RANGE_BITS / 2);
	double coef = (double)aver / root;
	sprintf(s, "%s, K = %f (including DP overhead)", names[Method], coef);
	std::cout << s << std::endl;
	if (RANGE_BITS < 40)
		std::cout << "Note: RANGE_BITS is too small to measure K precisely" << std::endl;
	if (POINTS_CNT < 500)
		std::cout << "Note: POINTS_CNT is too small to measure K precisely" << std::endl;

	if (Method == METHOD_ADVANCED)
	{
		u64 mul = 1;
		for (int i = 0; i < MD_CNT; i++)
		{
			for (int j = 0; j < MD_LEN; j++)
				if (loop_stats[i][j])
				{
					sprintf(s, "loop size %llu: %llu", j * mul, loop_stats[i][j]);
					std::cout << s << std::endl;
				}
			mul *= MD_LEN;
		}
#ifdef ESCAPE_FROM_LARGE_LOOPS
		sprintf(s, "large_loop_cnt (for all kangs and points): %llu, large_loop_cnt_per_kang: %.2f", large_loop_cnt, ((double)large_loop_cnt) / (KANG_CNT * CPU_THR_CNT));
		std::cout << s << std::endl;
#endif
		sprintf(s, "L2 Size-2 cnt: %d", L2loop_cnt);
		std::cout << s << std::endl;
	}
}

int main()
{
	InitEc();
	sprintf(s, "Range: %d bits. DP: %d bits. Kangaroos: %d. Threads: %d. Points in test: %d. JMP_CNT: %d, JMP_CNT2: %d", RANGE_BITS, DP_BITS, KANG_CNT, CPU_THR_CNT, POINTS_CNT, JMP_CNT, JMP_CNT2);

	//TestKangaroo(0);//METHOD_SIMPLE
	//Sota:
	TestKangaroo(1);//METHOD_ADVANCED

	return 0;//pow
}
