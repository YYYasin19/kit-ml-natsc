import pygame
import time
import random

# Some global parameters to define color, sizes etc.
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)
dis_width = 600
dis_height = 400
snake_block = 10 # size of a snake segment
snake_speed = 15 # speed

def Your_score(score):
    """Print the score on screen."""
    score_font = pygame.font.SysFont("comicsansms", 15)
    value = score_font.render("Your Score: " + str(score), True, yellow)
    dis.blit(value, [0, 0])
    pass
 
def message(msg, color):
    font_style = pygame.font.SysFont("bahnschrift", 25)
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [dis_width / 6, dis_height / 3])

def draw_our_snake(snake_block, snake_list):
    """Draw the snake."""
    for x in snake_list:
        pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block]) 
    
def generate_food_position():
    """Make random food position."""
    # @TODO: Make random position
    from random import randrange
    foodx = randrange(0, dis_width, snake_block)
    foody = randrange(0, dis_height, snake_block)
    print(f"{foodx=}{foody=}")
    return [foodx,foody]

def move_snake(snake_List,snake_Head,Length_of_snake, do_append = True):
    """Move the snake with new head"""
    if do_append:
        snake_List.append(snake_Head)
    
    if len(snake_List) > Length_of_snake:
        snake_List.pop(0)

def check_crash_walls(x1,y1):
    """Check if it hits the wall"""
    crash_detect = False
    # check wall collisions
    return (x1 == dis_width or y1 == dis_height or x1 < 0 or y1 < 0)

def check_crash_self(snake_List,snake_Head):
    """Check if snake crashed in itself."""
    found_crash = False
    # Check if the head is anywhere where the body is (snake excluding the last item which is the head)
    return snake_Head in snake_List[:-1]

def gameLoop():
    game_over = False
    game_close = False
 
    x1 = dis_width / 2
    y1 = dis_height / 2
 
    x1_change = 0
    y1_change = 0
 
    snake_List = []  # list of coordinates [[x1, y1], [x1, y1], ...]
    Length_of_snake = 1
 
    foodx,foody = generate_food_position()
 
    while not game_over:
 
        while game_close == True:
            dis.fill(blue)
            message("You Lost! Press C-Play Again or Q-Quit", red)
            Your_score(Length_of_snake - 1)
            pygame.display.update()
 
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        gameLoop()
 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and x1_change != snake_block:
                    x1_change = -snake_block
                    y1_change = 0
                elif event.key == pygame.K_RIGHT and x1_change != -snake_block:
                    x1_change = snake_block
                    y1_change = 0
                elif event.key == pygame.K_UP and y1_change != snake_block:
                    y1_change = -snake_block
                    x1_change = 0
                elif event.key == pygame.K_DOWN and y1_change != -snake_block:
                    y1_change = snake_block
                    x1_change = 0
 
        if check_crash_walls(x1,y1):
            game_close = True
        x1 += x1_change
        y1 += y1_change
        
        dis.fill(blue)
        pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block])
        
        snake_Head = [x1,y1] 
        move_snake(snake_List,snake_Head,Length_of_snake)
 
        if check_crash_self(snake_List,snake_Head):
            game_close = True
 
        draw_our_snake(snake_block, snake_List)
        Your_score(Length_of_snake - 1)
 
        pygame.display.update()
 
        if x1 == foodx and y1 == foody:
            foodx,foody = generate_food_position()
            Length_of_snake += 1
 
        clock.tick(snake_speed)
 
    pygame.quit()
    # quit()

run_game = True # please submit you answer with run_game = False
if run_game:
    pygame.init()
    dis = pygame.display.set_mode((dis_width, dis_height))
    pygame.display.set_caption('Snake Game')
    clock = pygame.time.Clock()
    gameLoop()