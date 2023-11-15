from direct.gui.OnscreenImage import OnscreenImage
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import render2d
from panda3d.core import PointLight, AmbientLight, DirectionalLight, CollisionPolygon, Point3, CollisionNode, \
    CollisionBox, CollisionHandlerPusher, CollisionRay, CollisionTraverser, Light, CardMaker, LPlane, \
    CollisionHandlerQueue, GeomVertexFormat, GeomVertexData, GeomVertexWriter, Geom, GeomTriangles, GeomNode, Material, \
    Plane, CollisionSphere
from panda3d.core import PandaNode, NodePath
from panda3d.core import Filename
from panda3d.core import KeyboardButton
from panda3d.core import Vec3
import panda3d as pd3
import pygame
import numpy as np
from panda3d.core import CollisionNode, CollisionPlane, CollisionHandlerEvent, BitMask32
from direct.interval.IntervalGlobal import Sequence, Func, Wait
from panda3d.core import CollisionTraverser, CollisionHandlerEvent
from panda3d.core import CollisionNode, CollisionSphere

def satFun(angle):
    if angle > 90:
        angle = 90
    elif angle < -90:
        angle = -90.
    return angle


class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Using joystick: {self.joystick.get_name()}")
        else:
            print("No joystick found.")
            self.joystick = None

        # Load the model
        self.globalClock = None
        self.model = self.loader.loadModel("hanger/X-Wing_kev.gltf")
        self.model.reparentTo(self.render)

        # Set the model position and scale
        self.model.setPos(0, 0, 0)
        self.model.setScale(5, 5, 5)

        self.env_array = []

        for i in range(200):
            # Load the model
            self.env = self.loader.loadModel("hanger/ressources.gltf")
            self.env.reparentTo(self.render)

            # Set the model position and scale
            self.env.setPos(-75, -200+(i*220), 10)
            self.env.setScale(6.25, 6.25, 6.25)

        y = self.env.getY()
        # Add lights to the scene
        self.setupLights()
        self.createSkyDome(0)

        self.load=1.0

        #self.camera.setPos(0, -10000, 2000)  # Set the camera position (x, y, z)
        self.camera.lookAt(self.model)  # Set the look-at point

        # Set the initial velocity as a Vec3
        self.velocity = Vec3(0, 0, 0)
        #
        # # Initialize the traverser.
        # self.cTrav = CollisionTraverser()
        #
        # # Initialize the handler.
        # self.collHandEvent = CollisionHandlerEvent()
        # self.collHandEvent.addInPattern('into-%in')
        # self.collHandEvent.addOutPattern('outof-%in')
        #
        # # Make a variable to store the unique collision string count.
        # self.collCount = 0
        #
        # sColl = self.initCollisionSphere(self.model,True)
        #
        # # Add the Pusher collision handler to the collision traverser.
        # self.cTrav.addCollider(sColl[0], self.collHandEvent)
        #
        # # Accept the events sent by the collisions.
        # self.accept('into-' + sColl[1], self.collide3)
        # self.accept('outof-' + sColl[1], self.collide4)
        # print(sColl[1])
        #
        # sColl = self.initCollisionSphere(self.model,True)

        self.taskMgr.add(self.moveAndRotateModelTask, "moveAndRotateModelTask")


    def moveAndRotateModelTask(self, task):
        # if self.model.getZ() < -220:
        #     self.velocity.setX(0)
        #     self.velocity.setY(0)
        #     self.velocity.setZ(0)

        # Update the model's position over time based on velocity
        time_elapsed = globalClock.getDt()  # Get the time elapsed since the last frame

        # Update the position based on velocity
        new_pos = self.model.getPos() + self.velocity * time_elapsed

        # Set the new position of the model
        self.model.setPos(new_pos)

        self.camera.setPos(Vec3(new_pos.getX(),new_pos.getY()-50,new_pos.getZ()+10))  # Set the camera position (x, y, z)
        self.camera.setHpr(0, -10, 0)  # Set the camera orientation (heading, pitch, roll)

        # Check joystick input
        if self.joystick:
            pygame.event.pump()  # Pump Pygame events
            # Use self.joystick.get_axis(axis_number) for joystick input
            self.updateVelocityX(self.joystick.get_axis(0))
            self.updateVelocityZ(self.joystick.get_axis(1))

            #Thrust
            self.updateVelocityY(self.joystick.get_axis(3))

        if self.joystick.get_axis(1) > 0.2 or self.joystick.get_axis(1) < -0.2 or self.joystick.get_axis(0) > 0.2 or self.joystick.get_axis(0) < -0.2:
            self.rotateModel(self.joystick.get_axis(1), self.joystick.get_axis(0))
        else:
            self.model.setHpr(-180, satFun(self.model.getP()/1.04), satFun(self.model.getR()/1.04))

        #print(self.model.getY()/20000)

        if (self.model.getY()/20000) >= self.load:
            self.updateSkyDome(self.model.getY())
            self.load += 1.0

        #self.taskMgr.add(self.checkCollisionsTask, "checkCollisionsTask")

        return task.cont  # Continue running the task on the next frame

    def updateVelocityX(self, value):
        # Update the velocity along the X-axis based on joystick input
        self.velocity.setX(value*10)

    def updateVelocityZ(self, value):
        # Update the velocity along the Y-axis based on joystick input
        self.velocity.setZ(value*10)

    def updateVelocityY(self, value):
        self.velocity.setY((-value)*1000)
    # def setupLights(self):
    #     ####################################################################
    #
    #     # Create Ambient Light
    #     ambientLight = AmbientLight('ambientLight')
    #     ambientLight.setColor((0.0, 0., 0., 1))
    #     ambientLightNP = self.render.attachNewNode(ambientLight)
    #     self.render.setLight(ambientLightNP)
    #
    #     # Directional light 01
    #     directionalLight = DirectionalLight('directionalLight')
    #     directionalLight.setColor((0.8, 0.2, 0.2, 1))
    #     directionalLightNP = self.render.attachNewNode(directionalLight)
    #     # This light is facing backwards, towards the camera.
    #     directionalLightNP.setHpr(180, -20, 0)
    #     self.render.setLight(directionalLightNP)
    #
    #     # Directional light 02
    #     directionalLight = DirectionalLight('directionalLight')
    #     directionalLight.setColor((0.2, 0.2, 0.8, 1))
    #     directionalLightNP = self.render.attachNewNode(directionalLight)
    #     # This light is facing forwards, away from the camera.
    #     directionalLightNP.setHpr(0, -20, 0)
    #     self.render.setLight(directionalLightNP)
    #
    #     #####################################################################
    #     # # Create a point light
    #     # plight = PointLight('plight')
    #     # plight.setColor((1, 1, 1, 1))
    #     # plnp = self.render.attachNewNode(plight)
    #     # plnp.setPos(0, 0, 100)
    #     # self.render.setLight(plnp)
    #     #
    #     # # Create an ambient light
    #     # alight = AmbientLight('alight')
    #     # alight.setColor((0.2, 0.2, 0.2, 1))
    #     # alnp = self.render.attachNewNode(alight)
    #     # self.render.setLight(alnp)
    def setupLights(self, temperature=None):
        ####################################################################

        #Create Ambient Light

        ambientLight = AmbientLight('ambientLight')
        ambientLight.setColor((0., 0., 0., 0.))
        ambientLightNP = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNP)

        # Directional light 01
        pointLight = DirectionalLight('pointLight')
        #directionalLight.setColor((0.8, 0.2, 0.2, 1))
        pointLight.setColorTemperature(6000.)
        pointLightNP = self.render.attachNewNode(pointLight)
        pointLightNP.setHpr(0, 20, 0)
        pointLightNP.setPos(0, 0, -500)  # Position the light below the model
        self.render.setLight(pointLightNP)

        # # Directional light 02
        # directionalLight = DirectionalLight('directionalLight')
        # directionalLight.setColor((0.2, 0.2, 0.8, 1))
        # directionalLightNP = self.render.attachNewNode(directionalLight)
        # # This light is facing forwards, away from the camera.
        # directionalLightNP.setHpr(0, -20, 0)
        # self.render.setLight(directionalLightNP)

        #####################################################################
        # # Create a point light
        # plight = PointLight('plight')
        # plight.setColor((1, 1, 1, 1))
        # plnp = self.render.attachNewNode(plight)
        # plnp.setPos(0, 0, 100)
        # self.render.setLight(plnp)
        #
        # # Create an ambient light
        # alight = AmbientLight('alight')
        # alight.setColor((0.2, 0.2, 0.2, 1))
        # alnp = self.render.attachNewNode(alight)
        # self.render.setLight(alnp)
    def rotateModel(self, pitch, roll):
        # Rotate the model based on joystick input
        # Scale the input to control the rotation speed
        pitch += .10
        roll += .10

        self.model.setHpr(-180, satFun(self.model.getP() - pitch), satFun(self.model.getR() - roll))


    def createSkyDome(self,y):
        # Load a texture for the sky
        sky_texture = self.loader.loadTexture("hanger/space.jpg")

        # Create a CardMaker to generate a textured card
        cm = CardMaker("sky")
        cm.setFrameFullscreenQuad()

        # Create a node for the sky and attach it to the render
        self.sky_node = self.render.attachNewNode(cm.generate())
        self.sky_node.setTexture(sky_texture)
        # Set the sky_node to be at a large distance to ensure it surrounds the entire scene
        self.sky_node.setPos(0, 80000+y, 4500)
        self.sky_node.setScale(50000)  # Scale as needed based on your scene size

        self.sky_node.setTransparency(True)
        self.sky_node.setAlphaScale(0.95)

        self.sky_node.setLightOff()

    def updateSkyDome(self, y):

        self.sky_node.hide()

        # Load a texture for the sky
        sky_texture = self.loader.loadTexture("hanger/space.jpg")

        # Create a CardMaker to generate a textured card
        cm = CardMaker("sky")
        cm.setFrameFullscreenQuad()

        # Create a node for the sky and attach it to the render
        self.sky_node = self.render.attachNewNode(cm.generate())
        self.sky_node.setTexture(sky_texture)
        # Set the sky_node to be at a large distance to ensure it surrounds the entire scene
        self.sky_node.setPos(0, 80000 + y, 4500)
        self.sky_node.setScale(50000)  # Scale as needed based on your scene size

        self.sky_node.setTransparency(True)
        self.sky_node.setAlphaScale(0.95)

        self.sky_node.setLightOff()

    def createCollisionPlane(self):
        # Create a CollisionNode for the plane
        plane_col_node = CollisionNode("collision_plane")

        # Create an LPlanef representing a plane facing up at z=0
        plane = CollisionPlane(Plane(Vec3(0, 0, 1), Point3(0, 0, 0)))

        # Add the plane to the CollisionNode
        plane_col_node.addSolid(CollisionPlane(plane))
        self.plane_collider = self.env.attachNewNode(plane_col_node)

        # Set the collision mask for the plane
        self.plane_collider.setCollideMask(BitMask32.bit(1))

    # def checkCollisionsTask(self, task):
    #     # Create a CollisionTraverser for checking collisions
    #     traverser = CollisionTraverser()
    #
    #     # Create a CollisionHandlerQueue to store collision entries
    #     queue = CollisionHandlerQueue()
    #     queue = CollisionHandlerQueue()
    #
    #     # Check if both colliders are not empty and have collision nodes
    #     if not self.plane_collider.is_empty() and self.plane_collider.node().is_collision_node():
    #         # Add the collision plane to the traverser
    #         traverser.addCollider(self.plane_collider, queue)
    #
    #     if not self.env.is_empty() and self.env.node().is_collision_node():
    #         # Add the model to the traverser (assuming self.model is the root of your collision hierarchy)
    #         traverser.addCollider(self.env, queue)
    #
    #     # Traverse the scene to detect collisions
    #     traverser.traverse(self.render)
    #
    #     # Check for collisions in the handler queue
    #     for entry in queue.entries:
    #         # Perform actions when a collision occurs
    #         print("Collision detected with the plane!")
    #
    #     return task.cont
    #
    #
    # def createPlaneVisual(self, plane, size=10, color=(255., 0, 0, 1)):
    #     card_maker = CardMaker("plane_visual")
    #     card_maker.setFrame(-size, size, -size, size)
    #
    #     # Create a NodePath for the visual representation
    #     plane_visual = NodePath(card_maker.generate())
    #     plane_visual.lookAt(0, 0, 1)  # Orient the plane to face the camera
    #
    #     # Position the plane at the specified distance along its normal
    #     plane_visual.setPos(plane_visual, 0, 0, plane[3])
    #
    #     # Set the color of the visual representation
    #     material = Material()
    #     material.setDiffuse((color[0], color[1], color[2], color[3]))
    #     plane_visual.setMaterial(material, 1)
    #
    #     return plane_visual



app = MyApp()
app.run()






