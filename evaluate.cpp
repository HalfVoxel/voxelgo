#include <vector>
#include <iostream>
#include <map>
#include <array>
#include <cmath>
#include <sstream>
#include <cassert>
#include <set>
#include <stack>

#include "net.h"

using namespace std;

typedef vector<float> vf;
typedef vector<vf> vff;
typedef vector<vff> vfff;
typedef vector<vfff> vffff;
typedef pair<int,int> pii;

#define itervfff(i, j, k, tensor, pad) for (int i = pad; i < (int)tensor.size() - pad; i++) \
		for (int j = pad; j < (int)tensor[0].size() - pad; j++) \
		for (int k = 0; k < (int)tensor[0][0].size(); k++)




void parse(stringstream &input, vf &output) {
	int dim;
	input >> dim;
	
	assert(dim == 1);
	assert(output.size() == 0);

	int d1;
	input >> d1;

	for (int i = 0; i < d1; i++) {
		float v;
		input >> v;
		output.push_back(v);
	}
}

void parse(stringstream &input, vffff &output) {
	int dim;
	input >> dim;
	
	assert(dim == 4);
	assert(output.size() == 0);

	int d1, d2, d3, d4;
	input >> d1 >> d2 >> d3 >> d4;

	for (int i = 0; i < d1; i++) {
		vfff partial1;
		for (int j = 0; j < d2; j++) {
			vff partial2;
			for (int k = 0; k < d3; k++) {
				vf partial3;
				for (int q = 0; q < d4; q++) {
					float v;
					input >> v;
					partial3.push_back(v);
				}
				partial2.push_back(partial3);
			}
			partial1.push_back(partial2);
		}
		output.push_back(partial1);
	}
}

int padding = 2;
vfff conv2d(const vfff &input, const vffff &filter, const vf &strides) {
	// Zeroed output array
	vfff output = vfff(input.size(), vff(input[0].size(), vf(filter[0][0][0].size(), 0)));

	for (int i = 0; i < output.size() - 2*padding; i++) {
		for (int j = 0; j < output[0].size() - 2*padding; j++) {
			for (int k = 0; k < output[0][0].size(); k++) {
				float acc = 0;

				for (int di = 0; di < filter.size(); di++) {
					for (int dj = 0; dj < filter[0].size(); dj++) {
						for (int q = 0; q < filter[0][0].size(); q++) {
							acc += input[strides[1] * i + di][strides[2] * j + dj][q] * filter[di][dj][q][k];
						}
					}
				}

				output[i + padding][j + padding][k] = acc;
			}
		}
	}

	return output;
}

void add(vfff &data, vf &val) {
	itervfff(i, j, k, data, padding) {
		data[i][j][k] += val[k];
	}
}

void div(vfff &data, float val) {
	itervfff(i, j, k, data, padding) {
		data[i][j][k] /= val;
	}
}

void relu(vfff &data) {
	itervfff(i, j, k, data, padding) {
		data[i][j][k] = max(data[i][j][k], 0.0f);
	}
}

float sum(const vfff &data) {
	float acc = 0;
	itervfff(i, j, k, data, padding) {
		acc += data[i][j][k];
	}
	return acc;
}

void normalize(vfff &data) {
	div(data, sum(data));
}

void softmax(vfff &data) {
	itervfff(i, j, k, data, padding) {
		data[i][j][k] = exp(data[i][j][k]);
	}

	normalize(data);
}

pii argmax(const vfff &data) {
	pii best;
	float best_val = -10000000;
	itervfff(y, x, k, data, padding) {
		float v = data[y][x][k];

		if (v > best_val) {
			best_val = v;
			best = pii(x - padding, y - padding);
		}
	}

	return best;
}

vf strides = { 1, 1, 1, 1 };

vffff W_conv1, W_conv2, W_conv3, W_conv4;
vf b_conv1, b_conv2, b_conv3, b_conv4;

int our_id = -1;
int opponend_id = -1;

const int OutOfBounds = -2;
const int KoBlocked = -1;
const int Empty = 0;

void parse () {
	stringstream ss;

	ss.str(W_conv1_data);
	parse(ss, W_conv1);

	ss.str(b_conv1_data);
	parse(ss, b_conv1);

	ss.str(W_conv2_data);
	parse(ss, W_conv2);

	ss.str(b_conv2_data);
	parse(ss, b_conv2);

	ss.str(W_conv3_data);
	parse(ss, W_conv3);

	ss.str(b_conv3_data);
	parse(ss, b_conv3);

	ss.str(W_conv4_data);
	parse(ss, W_conv4);

	ss.str(b_conv4_data);
	parse(ss, b_conv4);
}

vff read_board_from_stdin() {
	string s;
	cin >> s;
	stringstream ss;
	ss.str(s);

	vff board;
	string str;
	for (int y = 0; y < 19; y++) {
		vf partial;
		for (int x = 0; x < 19; x++) {
			getline(ss, str, ',');
			int v = stoi(str);
			partial.push_back(v);
		}
		board.push_back(partial);
	}
	return board;
}

int dx[] = { 1, 0, -1, 0 };
int dy[] = { 0, 1, 0, -1 };

int get (const vff &board, int x, int y) {
	if (x < 0 || y < 0 || x >= 19 || y >= 19) {
		// Out of bounds
		return OutOfBounds;
	}
	return board[y][x];
}

pair<int, set<pii>> freedoms (const vff &board, int x, int y) {
	auto color = get(board, x, y);
	set<pii> seen;

	if (color == 0 || color == KoBlocked) {
		return make_pair(0, seen);
	}
	
	seen.insert(pii(x,y));
	set<pii> seen_freedoms;
	stack<pii> st;
	st.push(pii(x,y));
	int freedoms = 0;

	while (!st.empty()) {
		auto p = st.top();
		st.pop();

		for (int i = 0; i < 4; i++) {
			auto po = pii(p.first + dx[i], p.second + dy[i]);

			auto other = get(board, po.first, po.second);
			if (other == Empty || other == KoBlocked) {
				if (seen_freedoms.find(po) == seen_freedoms.end()) {
					seen_freedoms.insert(po);
					freedoms++;
				}
			} else if (other == color) {
				if (seen.find(po) == seen.end()) {
					seen.insert(po);
					st.push(po);
				}
			}
		}
	}

	return make_pair(freedoms, seen);
}

void remove_if_taken(vff &board, int x, int y) {
	auto info = freedoms(board, x, y);
	if (info.first == 0) {
		// Remove all
		for (auto p : info.second) {
			board[p.second][p.first] = 0;
		}
	}
}

bool make_move (vff &board, int x, int y) {
	if (get(board, x, y) != 0) {
		return false;
	}

	board[y][x] = our_id;

	for (int i = 0; i < 4; i++) {
		int nx = x + dx[i];
		int ny = y + dy[i];
		if (get(board, nx, ny) == opponend_id) {
			remove_if_taken(board, nx, ny);
		}
	}

	for (int i = 0; i < 4; i++) {
		int nx = x + dx[i];
		int ny = y + dy[i];
		if (get(board, nx, ny) == our_id) {
			remove_if_taken(board, nx, ny);
		}
	}

	remove_if_taken(board, x, y);

	return get(board, x, y) == our_id;
}

vff connected_components(const vff &board) {
	vff comps = vff(19, vf(19, -1));

	int id = 0;
	for (int x = 0; x < 19; x++) {
		for (int y = 0; y < 19; y++) {
			if (comps[y][x] == -1 && get(board, x, y) == our_id) {
				auto info = freedoms(board, x, y);

				for (auto p : info.second) {
					comps[p.second][p.first] = id;
				}
				assert(info.first != 0);
				id++;
			}
		}
	}

	return comps;
}

bool is_our_eye(const vff &board, const vff &comps, int x, int y) {
	int direct = 0;
	int outside_diagonals = 0;
	int outside_direct = 0;
	int diagonals = 0;
	bool different_groups = false;
	int groups_id = -1;
	for (int dx = -1; dx <= 1; dx++) {
		for (int dy = -1; dy <= 1; dy++) {
			if (dx != 0 || dy != 0) {
				int nx = x + dx;
				int ny = y + dy;
				if (get(board, nx, ny) == our_id) {
					if (groups_id == -1) {
						groups_id = comps[ny][nx];
					} else if (comps[ny][nx] != groups_id) {
						different_groups = true;
					}

					if (dx != 0 && dy != 0) {
						diagonals++;
					} else {
						direct++;
					}
				} else if (get(board, nx, ny) == OutOfBounds) {

					if (dx != 0 && dy != 0) {
						outside_diagonals++;
					} else {
						outside_direct++;
					}
				}
			}
		}
	}

	int outside = outside_direct + outside_diagonals;

	// Definitely qualifies as an eye
	if (outside == 0) {
		if (different_groups) {
			// We can skip 1 on the diagonals (todo: this will never be true...?)
			return direct == 4 && diagonals >= 3;
		} else {
			// Only a single group surrounding it, we can skip the diagonals
			return direct == 4;
		}
	} else {
		if (different_groups) {
			// We can skip 1 on the diagonals (todo: this will never be true...?)
			return direct + diagonals + outside >= 8;
		} else {
			// Only a single group surrounding it, we can skip the diagonals
			return direct + outside_direct == 4;
		}
	}
}

bool valid_move(const vff &board, const vff &comps, int x, int y) {
	// Prevent placing in our own eyes
	if (is_our_eye(board, comps, x, y)) {
		return false;
	}

	auto dup = board;
	return make_move(dup, x, y);
}

vfff convert_board_to_input(const vff &board) {
	vfff input = vfff(19 + padding*2, vff(19 + padding*2, vf(5, 0)));

	for (int y = 0; y < 19; y++) {
		for (int x = 0; x < 19; x++) {
			input[y + padding][x + padding][3] = (x/(19.0f - 1)) - 0.5f;
			input[y + padding][x + padding][4] = (y/(19.0f - 1)) - 0.5f;

			if (board[y][x] == our_id) {
				input[y + padding][x + padding][0] = 1;
			} else if (board[y][x] == opponend_id) {
				input[y + padding][x + padding][1] = 1;
			} else {
				// Empty
			}

			input[y + padding][x + padding][2] = freedoms(board, x, y).first;
		}
	}

	return input;
}

vfff eval_net(const vfff &input) {
	#ifdef DEBUG
	cerr << "INPUT: " << endl;
	itervfff(i, j, k, input, padding) {
		if (k == 0) {
			cerr << input[i][j][k] << " ";
		}
	}
	cerr << endl;
	#endif

	assert(input.size() == 19 + 2*padding);
	assert(input[0].size() == 19 + 2*padding);
	assert(input[0][0].size() == 5);

	// Layer 1
	auto last = conv2d(input, W_conv1, strides);
	add(last, b_conv1);
	relu(last);

	

	// Layer 2
	last = conv2d(last, W_conv2, strides);
	add(last, b_conv2);
	relu(last);

	// Layer 3
	last = conv2d(last, W_conv3, strides);
	add(last, b_conv3);
	relu(last);

	// Layer 4
	last = conv2d(last, W_conv4, strides);
	add(last, b_conv4);
	relu(last);

	// Layer 5
	softmax(last);

	assert(last.size() == 19 + 2*padding);
	assert(last[0].size() == 19 + 2*padding);
	assert(last[0][0].size() == 1);

	#ifdef DEBUG
	cerr << "OUTPUT: " << endl;
	itervfff(i, j, k, input, padding) {
		if (k == 0) {
			cerr << last[i][j][k] << " ";
		}
	}
	cerr << endl;
	#endif

	return last;
}


pair<int,int> run(const vff &board) {
	vfff input = convert_board_to_input(board);

	auto last = eval_net(input);

#ifdef DEBUG
	cerr << "print" << endl;
	cerr << our_id << " " << opponend_id << endl;

	cerr << "input" << endl;
	itervfff(i, j, k, input, padding) {
		cerr << input[i][j][k] << " ";
	}
	cerr << endl;

	cerr << "board" << endl;
	for (int y = 0; y < 19; y++) {
		for (int x = 0; x < 19; x++) {
			if (board[y][x] == our_id) {
				cerr << 1;
			} else if (board[y][x] == opponend_id) {
				cerr << -1;
			} else {
				cerr << board[y][x];
			}
			cerr << " ";
		}
	}
	cerr << endl;

	cerr << "visualize" << endl;
	itervfff(i, j, k, last, padding) {
		cerr << last[i][j][k] << " ";
	}
	cerr << endl;
#endif

	auto comps = connected_components(board);

	for (int i = 0; i < 19*19; i++) {
		auto candidate = argmax(last);

		if (valid_move(board, comps, candidate.first, candidate.second)) {
			return candidate;
		} else {
			cerr << "Invalid move, selecting next candidate..." << endl;
			// Set that to zero
			last[candidate.second + padding][candidate.first + padding][0] = 0;
			assert(argmax(last) != candidate);
		}
	}

	cerr << "Found no valid move" << endl;
	return pii(-1,-1);
}

vff empty_board() {
	return vff(19, vf(19, Empty));
}

int main () {
	parse();

	vff current_board;

	string command;
	while(cin >> command) {
		if (command == "dummy") {
			current_board = empty_board();
		} else if (command == "settings") {
			string type;
			cin >> type;
			if (type == "your_botid") {
				cin >> our_id;
				opponend_id = our_id == 1 ? 2 : 1;
			} else if (type == "player_names") {
				string a;
				cin >> a;
			} else {
				string a;
				cin >> a;
			}
		} else if (command == "update") {
			string type;
			cin >> type;
			if (type == "game") {
				string subtype;
				cin >> subtype;
				if (subtype == "field") {
					current_board = read_board_from_stdin();
				} else {
					// Skip
					int dummy;
					cin >> dummy;
				}
			} else {
				string player = type;

				string update_type;
				cin >> update_type;
				if (update_type == "last_move") {
					string subtype;
					cin >> subtype;
					if (subtype == "pass") {

					} else if (subtype == "place_move") {
						int x, y;
						cin >> x >> y;
					} else {
						assert(subtype == "Null");
					}
				} else {
					float points;
					cin >> points;
				}
			}
		} else if (command == "action") {
			string type;
			cin >> type;
			assert(type == "move");
			float t;
			cin >> t;

			auto move = run(current_board);
			if (move.first == -1) {
				cout << "pass" << endl;
			} else {
				cout << "place_move " << move.first << " " << move.second << endl;
			}
		} else if (command == "Round") {
			// Only in debug input
			int round;
			cin >> round;
		} else if (command == "Output") {
			string s;
			cin >> s >> s >> s >> s;
			if (s == "\"place_move") {
				cin >> s >> s;
			}
		} else {
			cerr << "Could not parse command '" << command << "'" << endl;
			exit(1);
		}
	}
}