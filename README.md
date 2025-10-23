---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

trained my first PPO policy on google colab and loaded it locally for rendering on MuJoCo

went well for a start, my UR5e robot end-effector reached it's target every now and then

will tune my PPO policy to get better result, i need my robot to be able to this as easy as ABC

here's a clip of when it did reach it's target(feels like magic...look at my baby go!):



https://github.com/user-attachments/assets/f8b89be4-75cb-4b2b-ab06-5cca39fb8830 <img width="1366" height="768" alt="screenshot" src="https://github.com/user-attachments/assets/d0ab93f8-3805-47db-ac5a-f2526b8518df" />

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


tuned my robot PPO policy by simply decreasing the learning rate to enable stability and increasing the batch size for better gradient estimates

it's crazy how those slight tweaks made the UR5e arm learn better and faster

just look at my baby go! could I make it reach faster? probably but atm i'm satisfied



https://github.com/user-attachments/assets/b799451e-47e7-4489-8c09-a20861f94eb0
<img width="1366" height="768" alt="Screenshot From 2025-10-23 09-55-10" src="https://github.com/user-attachments/assets/a1331330-60b3-4719-9fd5-031e6a3ff2e5" />
<img width="1280" height="1024" alt="Screenshot From 2025-10-23 09-55-18" src="https://github.com/user-attachments/assets/980eecfd-2a9f-4539-828b-89785b3b6ab4" />

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



