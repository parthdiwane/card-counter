#include <stdio.h>
#include <vector>
#include <numeric>
#include "card_helpers.h"
using namespace std;
auto remaining = map<int, int>();

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
    remaining = remaining_deck_counts(played /*seen*/, /*num_decks=*/1);

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
    if(remaining_cards > 0) bust_if_hit = static_cast<double>(bust_count) / remaining_cards;

    //dealer calcs
    int dealer_needed = 22 - dealer_hand; // any card >= needed busts
    //sum of probability of dealer not busting if they hit(if they must hit)
    //recursive function that calculates dealer probability at every single branch 

    std::map <int, double> dealer_dist = dealerRecurse(dealer_hand, remaining, remaining_cards);
    
    
    


    // Placeholder decision: keep original simple rule (hit if player's last-two sum < 17)
    // Player/dealer stats are now available for more advanced strategies.
    return player_hand < 17;
}

int main() {
    //claude code tests
    // printf("=== Card Counter Algorithm Tests ===\n\n");

    // // Test 1: Full deck counts
    // printf("Test 1 - Full deck counts (1 deck):\n");
    // std::vector<int> empty;
    // auto full_deck = remaining_deck_counts(empty, 1);
    // int total_cards = 0;
    // for (const auto& p : full_deck) {
    //     if (p.second > 0) {
    //         printf("  Card %2d: %d\n", p.first, p.second);
    //         total_cards += p.second;
    //     }
    // }
    // printf("  Total cards: %d (expected 52)\n", total_cards);

    // // Test 2: Remaining deck after some cards played
    // printf("\nTest 2 - Remaining deck after playing [10, 5, 10, 7]:\n");
    // std::vector<int> played = {10, 5, 10, 7};
    // auto remaining_test = remaining_deck_counts(played, 1);
    // printf("  Card 5:  %d (expected 3)\n", remaining_test[5]);
    // printf("  Card 7:  %d (expected 3)\n", remaining_test[7]);
    // printf("  Card 10: %d (expected 14)\n", remaining_test[10]);

    // // Test 3: Player bust probability calculation
    // printf("\nTest 3 - Player bust probability:\n");
    // // Player with hand = 12, needs 10+ to bust
    // int player_hand = 12;
    // int needed = 22 - player_hand;  // 10
    // int bust_count = 0;
    // int remaining_cards = 0;
    // for (const auto& p : full_deck) {
    //     remaining_cards += p.second;
    //     if (p.first >= needed) {
    //         bust_count += p.second;
    //     }
    // }
    // double bust_prob = static_cast<double>(bust_count) / remaining_cards;
    // printf("  Player hand: %d, cards that bust: %d, total: %d\n", player_hand, bust_count, remaining_cards);
    // printf("  Bust probability: %.4f (expected ~0.31 for 10s only)\n", bust_prob);

    // // Test 4: Dealer recursion - dealer showing 6 (worst card for dealer)
    // printf("\nTest 4 - Dealer distribution starting at 6:\n");
    // auto deck = remaining_deck_counts(empty, 1);
    // int deck_size = 52;
    // auto dist = dealerRecurse(6, deck, deck_size);
    // double total_prob = 0.0;
    // for (const auto& p : dist) {
    //     printf("  Outcome %2d: %.4f\n", p.first, p.second);
    //     total_prob += p.second;
    // }
    // printf("  Total probability: %.4f (should be ~1.0)\n", total_prob);

    // // Test 5: Dealer recursion - dealer showing 10 (strong card)
    // printf("\nTest 5 - Dealer distribution starting at 10:\n");
    // deck = remaining_deck_counts(empty, 1);
    // dist = dealerRecurse(10, deck, deck_size);
    // total_prob = 0.0;
    // for (const auto& p : dist) {
    //     printf("  Outcome %2d: %.4f\n", p.first, p.second);
    //     total_prob += p.second;
    // }
    // printf("  Total probability: %.4f (should be ~1.0)\n", total_prob);

    // // Test 6: Dealer already at 17 (should stand immediately)
    // printf("\nTest 6 - Dealer at 17 (should stand):\n");
    // deck = remaining_deck_counts(empty, 1);
    // dist = dealerRecurse(17, deck, deck_size);
    // printf("  Outcome 17: %.4f (expected 1.0)\n", dist[17]);

    // // Test 7: Dealer at 22 (already busted)
    // printf("\nTest 7 - Dealer at 22 (already busted):\n");
    // deck = remaining_deck_counts(empty, 1);
    // dist = dealerRecurse(22, deck, deck_size);
    // printf("  Outcome 22: %.4f (expected 1.0)\n", dist[22]);

    // printf("\n=== Tests Complete ===\n");
    // return 0;
}
