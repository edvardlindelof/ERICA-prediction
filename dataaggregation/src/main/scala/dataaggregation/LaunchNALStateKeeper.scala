package dataaggregation

import akka.actor._
import org.joda.time.DateTime
import com.github.tototoshi.csv.CSVWriter

import sp.EricaEventLogger._
import sp.gPubSub.API_Data.EricaEvent

import scala.collection.mutable
import scala.util.Try
import java.io.File

object LaunchNALStateKeeper extends App {
  val system = ActorSystem("DataAggregation")

  val evHandler = new NALStateKeeper
  system.actorOf(Props(new Logger(evHandler)), "EricaEventLogger")
}

class NALStateKeeper extends RecoveredEventHandler {

  implicit def stringToDateTime(s: String) = DateTime.parse(s)

  val samplingIntervalMins = 23
  var nextSampleTime: DateTime = null

  val timeOfFirstEvent = mutable.Map[Int, DateTime]() // CareContactId -> DateTime

  val priorities = List("PRIO1", "PRIO2", "PRIO3", "PRIO4", "PRIO5")
  val teams = List("NAKME", "NAKM", "NAKKI", "NAKOR", "NAKBA", "NAKÖN") // "valid clinics" according to ericaBackend code
  val invalidTeams = List("NAKIN", "NAK23T", "NAK29", "NAKKK") // TODO should I include these??
  val patientKinds = List("all", "MEP", "triaged", "metdoctor", "done") ::: priorities ::: teams ::: invalidTeams
  val patientSets = patientKinds.map(str => str -> mutable.Set[Int]()).toMap // sets are of CareContactIds

  val minsToSaveRollingValues = 200 // needs to be longer than longest rolling time bco timediff between latest event and nextSampleTime
  val rollingValueKinds = "ttt" :: "ttl" :: "ttk" :: Nil
  // Map["ttX", ListBuffer[(time of X-event, seconds to X-event)]]
  val rollingValueLBs = mutable.Map(rollingValueKinds.map(str => str -> mutable.ListBuffer[(DateTime, Int)]()):_*)

  val recentlyRemoved = mutable.ListBuffer.fill(500)(-1) // work-around bco events occurring after patient removal, ugh

  override def handleEvent(ev: EricaEvent): Unit = {

    if(nextSampleTime == null) nextSampleTime = ev.Start.plusMinutes(samplingIntervalMins)
    else if(ev.Start.isAfter(nextSampleTime)) {
      // making sure to let lists decide the order of things
      val timeList = Map("epochseconds" -> (nextSampleTime.getMillis / 1000).toInt).toList
      val rolling30s = rollingValueKinds.map{ kind =>
        val valuesFromLast30Mins = rollingValueLBs(kind).filter(_._1.isAfter(nextSampleTime.minusMinutes(30))).map(_._2)
        val avg = Try(valuesFromLast30Mins.sum / valuesFromLast30Mins.length).getOrElse(0)
        kind + "30" -> avg
      }
      val patientsSetsList = patientKinds.map(kind => kind -> patientSets(kind).size)

      //println(nextSampleTime + " --- nPatients: " + timeOfFirstEvent.size + "; rolling30s: " + rolling30s)
      WriteToCSV(timeList ::: rolling30s ::: patientsSetsList)
      nextSampleTime = nextSampleTime.plusMinutes(samplingIntervalMins)
    }

    if(recentlyRemoved.contains(ev.CareContactId)) return () // if event occurs after patient removed, ignore it
    else if(!patientSets("all").contains(ev.CareContactId)) {
      timeOfFirstEvent += ev.CareContactId -> ev.Start
      patientSets("all").add(ev.CareContactId)
    }

    rollingValueKinds.foreach{ kind => // drop rolling values older than we care about
      val tooOld = (dt: DateTime) => ev.Start.minusMinutes(minsToSaveRollingValues).isAfter(dt)
      rollingValueLBs(kind) = rollingValueLBs(kind).dropWhile(t => tooOld(t._1))
    }
    //println(rollingValueLBs("ttt").map(_._2))
    //println(rollingValueLBs("ttl").map(_._2))
    //println(rollingValueLBs("ttk").map(_._2))

    ev.Category match {
      case "RemovedPatient" => {
        timeOfFirstEvent -= ev.CareContactId
        patientSets.foreach(t => t._2.remove(ev.CareContactId))
        recentlyRemoved += ev.CareContactId
        recentlyRemoved.trimStart(1)
      }
      case "Q" => patientSets("all").add(ev.CareContactId)
      //case "T" => patientSets("all").add(ev.CareContactId)
      //case "P" => patientSets(ev.Type).add(ev.CareContactId)
      case "ReasonForVisitUpdate" if ev.Value == "MEP" => patientSets("MEP").add(ev.CareContactId)
      case "TeamUpdate" => patientSets(ev.Value).add(ev.CareContactId)
      //case s: String if s.contains("removed") => patientsAtNAL.remove(ev.CareContactId)
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

}

object WriteToCSV {
  var calledYet = false
  var filename = ""
  def apply(data: List[(String, Int)]) = {
    if(!calledYet) {
      calledYet = true
      filename = "output/NALState" + DateTime.now() + ".csv"
      val csvWriter = CSVWriter.open(new File(filename))
      csvWriter.writeRow(data.map(_._1)) // fieldNames
      csvWriter.writeRow(data.map(_._2)) // values
      csvWriter.close()
    } else {
      val csvWriter = CSVWriter.open(new File(filename), append = true)
      csvWriter.writeRow(data.map(_._2))
      csvWriter.close()
    }
  }
}
