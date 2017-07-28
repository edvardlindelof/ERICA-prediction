package dataaggregation

import org.joda.time.DateTime
import sp.EricaEventLogger._
import sp.gPubSub.API_Data.EricaEvent

import scala.collection.mutable
import scala.util.Try

class RollingWaitTimeKeeper(rollingMins: Int = 30) extends StateKeeper {

  implicit def stringToDateTime(s: String) = DateTime.parse(s)

  val minsToSaveRollingValues = 200 // needs to be longer than longest rolling time bco timediff between latest event and nextSampleTime
  val rollingValueKinds = "ttt" :: "ttl" :: "ttk" :: Nil
  val rollingValueLBs = mutable.Map(rollingValueKinds.map(str => str -> mutable.ListBuffer[(DateTime, Int)]()):_*)

  val recentlyRemoved = mutable.ListBuffer.fill(500)(-1) // work-around bco events occurring after patient removal, ugh

  val timeOfFirstEvent = mutable.Map[Int, DateTime]() // CareContactId -> DateTime
  val patientKinds = List("all", "triaged", "metdoctor", "done")
  val patientSets = patientKinds.map(str => str -> mutable.Set[Int]()).toMap // sets are of CareContactIds

  override def handleEvent(ev: EricaEvent): Unit = {

    if(recentlyRemoved.contains(ev.CareContactId)) return () // if event occurs after patient removed, ignore it
    else if(!patientSets("all").contains(ev.CareContactId)) {
      timeOfFirstEvent += ev.CareContactId -> ev.Start
      patientSets("all").add(ev.CareContactId)
    }

    rollingValueKinds.foreach{ kind => // drop rolling values older than we care about
      val tooOld = (dt: DateTime) => ev.Start.minusMinutes(minsToSaveRollingValues).isAfter(dt)
      rollingValueLBs(kind) = rollingValueLBs(kind).dropWhile(t => tooOld(t._1))
    }
    ev.Category match {
      case "RemovedPatient" => {
        timeOfFirstEvent -= ev.CareContactId
        patientSets.foreach(t => t._2.remove(ev.CareContactId))
        recentlyRemoved += ev.CareContactId
        recentlyRemoved.trimStart(1)
      }
      case "Q" => patientSets("all").add(ev.CareContactId)
      case _ => ()
    }
    ev.Type match {
      case "TRIAGE" => {
        if(!patientSets("triaged").contains(ev.CareContactId)) { // if not already triaged
          val ttt = (ev.Start.getMillis - timeOfFirstEvent(ev.CareContactId).getMillis) / 1000
          rollingValueLBs("ttt") += ((ev.Start, ttt.toInt))
        }
        patientSets("triaged").add(ev.CareContactId)
      }
      case "LÄKARE" => {
        if(!patientSets("metdoctor").contains(ev.CareContactId)) { // if not already met doctor
          val ttl = (ev.Start.getMillis - timeOfFirstEvent(ev.CareContactId).getMillis) / 1000
          rollingValueLBs("ttl") += ((ev.Start, ttl.toInt))
        }
        patientSets("metdoctor").add(ev.CareContactId)
      }
      case "KLAR" => {
        if(!patientSets("done").contains(ev.CareContactId)) { // if not already done
          val ttl = (ev.Start.getMillis - timeOfFirstEvent(ev.CareContactId).getMillis) / 1000
          rollingValueLBs("ttk") += ((ev.Start, ttl.toInt))
        }
        patientSets("done").add(ev.CareContactId)
      }
      case _ => ()
    }
  }

  override def state(sampleTime: DateTime): List[(String, Int)] = {
    val rollingValues = rollingValueKinds.map { kind =>
      val valuesFromLastPeriod = rollingValueLBs(kind).filter(_._1.isAfter(sampleTime.minusMinutes(rollingMins))).map(_._2)
      val avg = Try(valuesFromLastPeriod.sum / valuesFromLastPeriod.length).getOrElse(0)
      kind + rollingMins -> avg
    }
    rollingValues
  }

}
