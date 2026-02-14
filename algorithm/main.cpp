#include <stdio.h>
#include <vector>
#include <map>
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

// Calculate expected value if player hits
// Inputs:
//   - player_hand: player's current total
//   - remaining: cards left in shoe {card_value: count}
//   - remaining_count: total cards left
//   - dealer_dist: dealer's outcome distribution
// Returns: EV of hitting (accounts for bust probability and recursive decisions)
double hittingEV(int player_hand, map<int, int>& remaining, int remaining_count,
                 const map<int, double>& dealer_dist) {
    // TODO: implement
    // Consider:
    //   - probability of busting with each draw
    //   - if not busting, recursively decide hit/stand with new hand
    //   - weight outcomes by draw probability
    return 0.0;
}


// Build game state from played cards
GameState buildGameState(const vector<int>& played, int num_decks = 1) {
    GameState state;

    // Get player and dealer hands
    HandStats player_stats = compute_hand_stats(played, true);
    HandStats dealer_stats = compute_hand_stats(played, false);

    state.player_hand = player_stats.last_two_sum;
    state.dealer_showing = dealer_stats.last_two_sum;

    // Calculate remaining cards
    state.remaining = remaining_deck_counts(played, num_decks);
    state.remaining_count = 0;
    for (const auto& p : state.remaining) {
        state.remaining_count += p.second;
    }

    // Get dealer outcome distribution
    state.dealer_dist = dealerRecurse(state.dealer_showing, state.remaining, state.remaining_count);

    return state;
}

// Decide whether to hit based on EV comparison
// Returns: true if player should hit, false if player should stand
bool shouldHit(const vector<int>& played, int num_decks = 1) {
    if (played.size() < 2) {
        return false;
    }

    GameState state = buildGameState(played, num_decks);

    // Edge cases
    if (state.player_hand == 0 || state.dealer_showing == 0) {
        return false;
    }
    if (state.player_hand >= 21) {
        return false;  // already 21 or busted
    }

    // Calculate EVs
    double ev_stand = standingEV(state.player_hand, state.dealer_dist);
    double ev_hit = hittingEV(state.player_hand, state.remaining, state.remaining_count, state.dealer_dist);

    // Hit if hitting EV is better than standing EV
    return ev_hit > ev_stand;
}



int main() {
    // Example usage:
    // vector<int> played = {10, 5, 10, 7};  // dealer: 10, 10  player: 5, 7
    // bool hit = shouldHit(played);
    // printf("Should hit: %s\n", hit ? "yes" : "no");

    return 0;
}
