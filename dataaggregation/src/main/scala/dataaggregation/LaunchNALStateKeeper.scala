package dataaggregation

import akka.actor._
import org.joda.time.DateTime

import sp.EricaEventLogger._
import sp.gPubSub.API_Data.EricaEvent

object LaunchNALStateKeeper extends App {
  val system = ActorSystem("DataAggregation")

  val evHandler = new NALStateKeeper
  system.actorOf(Props(new Logger(evHandler, TTLOfNextLowPrioPatient, "NALState")), "EricaEventLogger")
  //system.actorOf(Props(new Logger(StaffKeeper, TTLOfNextLowPrioPatient)), "EricaEventLogger")
  //system.actorOf(Props(new Logger()), "EricaEventLogger")
}

class NALStateKeeper extends StateKeeper {

  val rollingWaitTimeKeeper = new RollingWaitTimeKeeper()

  override def state(sampleTime: DateTime): List[(String, Int)] = {
    rollingWaitTimeKeeper.state(sampleTime) ::: PatientKindKeeper.state(sampleTime)
  }

  override def handleEvent(ev: EricaEvent): Unit = {
    rollingWaitTimeKeeper.handleEvent(ev)
    PatientKindKeeper.handleEvent(ev)
  }
}
