#include <stdio.h>
#include <vector>
#include <numeric>
#include "card_helpers.h"
using namespace std;
bool hit(vector<int> &played){
    // find player hand and dealer hand stats (segmented)
    if(played.size() < 2){
        return false;
    }

    HandStats player_stats = compute_hand_stats(played, /*for_player=*/true);
    HandStats dealer_stats = compute_hand_stats(played, /*for_player=*/false);

    // If either hand is empty, we don't decide to hit
    if(player_stats.last_two_sum == 0 || dealer_stats.last_two_sum == 0){
        return false;
    }

    int player_hand = player_stats.last_two_sum;
    int dealer_hand = dealer_stats.last_two_sum;

    // Compute remaining cards in the shoe (default 1 deck) and use that for probabilities
    auto remaining = remaining_deck_counts(played /*seen*/, /*num_decks=*/1);

    // Determine probability of player busting if they hit using remaining cards
    int needed = 22 - player_hand; // any card >= needed busts
    int bust_count = 0;
    int remaining_cards = 0;
    for(const auto &pair : remaining){
        int card_value = pair.first;
        int card_left = pair.second;
        remaining_cards += card_left;
        if(card_value >= needed){
            bust_count += card_left;
        }
    }

    double bust_if_hit = 0.0;
    if(remaining_cards > 0) bust_if_hit = (bust_count * 100.0) / remaining_cards;

    //dealer calcs
    int dealer_needed = 22 - dealer_hand; // any card >= needed busts
    //sum of probability of dealer not busting if they hit(if they must hit)
    int dealer_no_bust = 0;
    
    
    
    


    // Placeholder decision: keep original simple rule (hit if player's last-two sum < 17)
    // Player/dealer stats are now available for more advanced strategies.
    return player_hand < 17;
}
