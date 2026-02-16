#include <stdio.h>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <iostream>
#include "card_helpers.h"

using namespace std;

struct GameState {
    int player_hand;                   
    int dealer_showing;                
    map<int, int> remaining;            // cards left in shoe
    int remaining_count;                // total cards left
    map<int, double> dealer_dist;       // dealer's outcome distribution
};

// calcutes standing ev for the player
// inputs: player hand = current sum of players hand, dealer_dist is the key value mapping of the chance that the dealer has a hand from [17,22] 
// returns standing ev based on the current player hand and dealer distrabution
double standingEV(int player_hand, const map<int, double>& dealer_dist) {
    double ev = 0.0;
    for(const auto &[dealer_value, prob] : dealer_dist) {
        if(dealer_value > 21) {
            ev += prob;
        }
        else if(dealer_value < player_hand) {
            ev += (1 * prob);
        } else if(dealer_value > player_hand) {
            ev += (-1 * prob);
        }
    }
    return ev;
}
// calculates the ev for standing 
// inputs: player_hand = current sum for player hands, remaining mapping of {card value: count for card val}; remaining_count = number of cards remaining in the deck
// returns hitting ev 
    double hittingEV(int player_hand, map<int, int>& remaining, int remaining_count,
                    const map<int, double>& dealer_dist) {
        double total_ev = 0.0;
        for(auto& [card_value, count] : remaining) {
            if(count == 0) {
                continue;
            }
            double p_draw = count / (double) remaining_count;
            int new_hand = player_hand + card_value; 
            

            if(new_hand > 21) {
                total_ev += -1 * p_draw; // base case
            } else {
                count -= 1;
                remaining_count -= 1;


                double ev_stand = standingEV(new_hand, dealer_dist);
                double ev_hit = hittingEV(new_hand, remaining, remaining_count, dealer_dist); // backtrack

                total_ev += max(ev_hit, ev_stand) * p_draw;
                count += 1; // undo backtrack decison for the count and remaining count
                remaining_count += 1;
            }

        }
        return total_ev;
    }


GameState buildGameState(const vector<int>& played, int num_decks = 1) {
    GameState state;

    HandStats player_stats = compute_hand_stats(played, true);
    HandStats dealer_stats = compute_hand_stats(played, false);

    state.player_hand = player_stats.last_two_sum;
    state.dealer_showing = dealer_stats.last_two_sum;

    state.remaining = remaining_deck_counts(played, num_decks);
    state.remaining_count = 0;
    for (const auto& p : state.remaining) {
        state.remaining_count += p.second;
    }

    state.dealer_dist = dealerRecurse(state.dealer_showing, state.remaining, state.remaining_count);

    return state;
}


bool shouldHit(const vector<int>& played, int num_decks = 1) {
    if (played.size() < 2) {
        return false;
    }

    GameState state = buildGameState(played, num_decks);

    if (state.player_hand == 0 || state.dealer_showing == 0) {
        return false;
    }
    if (state.player_hand >= 21) {
        return false; 
    }

    double ev_stand = standingEV(state.player_hand, state.dealer_dist);
    double ev_hit = hittingEV(state.player_hand, state.remaining, state.remaining_count, state.dealer_dist);

    return ev_hit > ev_stand;
}



// Parse comma-separated integers from string (e.g., "10,5,6,7")
vector<int> parseCards(const string& input) {
    vector<int> cards;
    stringstream ss(input);
    string token;
    while (getline(ss, token, ',')) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t\n\r");
        size_t end = token.find_last_not_of(" \t\n\r");
        if (start != string::npos && end != string::npos) {
            token = token.substr(start, end - start + 1);
        }
        if (!token.empty()) {
            cards.push_back(stoi(token));
        }
    }
    return cards;
}

int main(int argc, char* argv[]) {
    // If --stdin flag is provided, read cards from stdin and output decision
    if (argc > 1 && string(argv[1]) == "--stdin") {
        string line;
        while (getline(cin, line)) {
            if (line.empty()) continue;

            vector<int> cards = parseCards(line);
            if (cards.size() < 4) {
                cout << "NEED_MORE_CARDS" << endl;
                continue;
            }

            bool hit = shouldHit(cards);
            cout << (hit ? "HIT" : "STAND") << endl;
        }
        return 0;
    }

    // If cards provided as argument, evaluate directly
    if (argc > 1 && string(argv[1]) != "--test") {
        vector<int> cards = parseCards(argv[1]);
        if (cards.size() < 4) {
            cout << "NEED_MORE_CARDS" << endl;
            return 1;
        }
        bool hit = shouldHit(cards);
        cout << (hit ? "HIT" : "STAND") << endl;
        return 0;
    }

    // Default: run test cases
    struct TestCase {
        vector<int> played;
        bool expected;
        const char* description;
    };

    vector<TestCase> tests = {
        // Hard stands (high hands, should never hit)
        {{10, 10, 10, 10}, false, "Player 20 vs dealer 20 - stand"},
        {{6, 9, 10, 10}, false, "Player 19 vs dealer 16 - stand"},
        {{10, 8, 10, 10}, false, "Player 18 vs dealer 20 - stand"},
        {{5, 10, 10, 10}, false, "Player 20 vs dealer 15 - stand"},
        {{9, 10, 10, 9}, false, "Player 19 vs dealer 19 - stand"},
        {{7, 10, 10, 10}, false, "Player 20 vs dealer 17 - stand"},

        // Player 17 vs various dealer upcards (should stand)
        {{10, 7, 10, 10}, false, "Player 17 vs dealer 20 - stand"},
        {{6, 7, 10, 10}, false, "Player 17 vs dealer 16 - stand"},
        {{2, 9, 10, 8}, false, "Player 18 vs dealer 11 - stand"},

        // Player 11 (should always hit, can't bust)
        {{6, 5, 10, 6}, true, "Player 11 vs dealer 16 - hit"},
        {{10, 4, 10, 7}, true, "Player 11 vs dealer 20 - hit"},
        {{9, 3, 10, 8}, true, "Player 11 vs dealer 19 - hit"},
        {{2, 6, 10, 5}, true, "Player 11 vs dealer 12 - hit"},

        // Player 10 (should always hit, can't bust)
        {{10, 4, 10, 6}, true, "Player 10 vs dealer 20 - hit"},
        {{5, 3, 10, 7}, true, "Player 10 vs dealer 15 - hit"},
        {{8, 4, 10, 6}, true, "Player 10 vs dealer 18 - hit"},

        // Player 9 (should always hit, can't bust)
        {{10, 5, 10, 4}, true, "Player 9 vs dealer 20 - hit"},
        {{6, 2, 10, 7}, true, "Player 9 vs dealer 16 - hit"},

        // Low hands (should always hit)
        {{10, 3, 10, 5}, true, "Player 8 vs dealer 20 - hit"},
        {{7, 2, 10, 5}, true, "Player 7 vs dealer 17 - hit"},
        {{10, 3, 10, 3}, true, "Player 6 vs dealer 20 - hit"},
        {{9, 2, 10, 4}, true, "Player 6 vs dealer 19 - hit"},

        // Player 16 (should hit against strong dealer)
        {{10, 8, 10, 8}, true, "Player 16 vs dealer 20 - hit"},
        {{9, 7, 10, 9}, true, "Player 16 vs dealer 19 - hit"},
        {{7, 6, 10, 10}, true, "Player 16 vs dealer 17 - hit"},

        // Player 15 (should hit against strong dealer)
        {{10, 6, 10, 9}, true, "Player 15 vs dealer 20 - hit"},
        {{9, 8, 10, 7}, true, "Player 15 vs dealer 19 - hit"},

        // Player 13-14 vs strong dealer (should hit)
        {{10, 6, 10, 7}, true, "Player 13 vs dealer 20 - hit"},
        {{10, 5, 10, 9}, true, "Player 14 vs dealer 20 - hit"},

        // Player 12 vs weak dealer
        {{2, 5, 10, 7}, true, "Player 12 vs dealer 12 - hit"},
    };

    int passed = 0;
    int failed = 0;

    for (int i = 0; i < (int)tests.size(); i++) {
        bool result = shouldHit(tests[i].played);
        if (result == tests[i].expected) {
            printf("PASSED test %d: %s\n", i + 1, tests[i].description);
            passed++;
        } else {
            printf("FAILED test %d: %s (expected %s, got %s)\n",
                   i + 1, tests[i].description,
                   tests[i].expected ? "hit" : "stand",
                   result ? "hit" : "stand");
            failed++;
        }
    }

    printf("\nResults: %d/%d passed\n", passed, passed + failed);
    return 0;
}